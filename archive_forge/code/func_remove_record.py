from __future__ import absolute_import, division, print_function
import traceback
from binascii import Error as binascii_error
from socket import error as socket_error
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def remove_record(self):
    result = {'changed': False, 'failed': False}
    if self.record_exists() == 0:
        return result
    if self.module.check_mode:
        self.module.exit_json(changed=True)
    update = dns.update.Update(self.zone, keyring=self.keyring, keyalgorithm=self.algorithm)
    update.delete(self.module.params['record'], self.module.params['type'])
    response = self.__do_update(update)
    self.dns_rc = dns.message.Message.rcode(response)
    if self.dns_rc != 0:
        result['failed'] = True
        result['msg'] = 'Failed to delete record (rc: %d)' % self.dns_rc
    else:
        result['changed'] = True
    return result