from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def make_configuration_request(self, body):
    if not self.check_mode:
        try:
            if self.syslog:
                if 'id' in body:
                    rc, result = request(self.url + 'storage-systems/{0}/syslog/{1}'.format(self.ssid, body['id']), method='POST', data=json.dumps(body), headers=HEADERS, **self.creds)
                else:
                    rc, result = request(self.url + 'storage-systems/{0}/syslog'.format(self.ssid), method='POST', data=json.dumps(body), headers=HEADERS, **self.creds)
                    body.update(result)
                if self.test:
                    self.test_configuration(body)
            elif 'id' in body:
                rc, result = request(self.url + 'storage-systems/{0}/syslog/{1}'.format(self.ssid, body['id']), method='DELETE', headers=HEADERS, **self.creds)
        except Exception as err:
            self.module.fail_json(msg='We failed to modify syslog configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))