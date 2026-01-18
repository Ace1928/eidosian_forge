from __future__ import absolute_import, division, print_function
import traceback
from binascii import Error as binascii_error
from socket import error as socket_error
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def modify_record(self):
    update = dns.update.Update(self.zone, keyring=self.keyring, keyalgorithm=self.algorithm)
    if self.module.params['type'].upper() == 'NS':
        query = dns.message.make_query(self.module.params['record'], self.module.params['type'])
        if self.keyring:
            query.use_tsig(keyring=self.keyring, algorithm=self.algorithm)
        try:
            if self.module.params['protocol'] == 'tcp':
                lookup = dns.query.tcp(query, self.module.params['server'], timeout=10, port=self.module.params['port'])
            else:
                lookup = dns.query.udp(query, self.module.params['server'], timeout=10, port=self.module.params['port'])
        except (dns.tsig.PeerBadKey, dns.tsig.PeerBadSignature) as e:
            self.module.fail_json(msg='TSIG update error (%s): %s' % (e.__class__.__name__, to_native(e)))
        except (socket_error, dns.exception.Timeout) as e:
            self.module.fail_json(msg='DNS server error: (%s): %s' % (e.__class__.__name__, to_native(e)))
        entries_to_remove = [n.to_text() for n in lookup.answer[0].items if n.to_text() not in self.value]
    else:
        update.delete(self.module.params['record'], self.module.params['type'])
    for entry in self.value:
        try:
            update.add(self.module.params['record'], self.module.params['ttl'], self.module.params['type'], entry)
        except AttributeError:
            self.module.fail_json(msg='value needed when state=present')
        except dns.exception.SyntaxError:
            self.module.fail_json(msg='Invalid/malformed value')
    if self.module.params['type'].upper() == 'NS':
        for entry in entries_to_remove:
            update.delete(self.module.params['record'], self.module.params['type'], entry)
    response = self.__do_update(update)
    return dns.message.Message.rcode(response)