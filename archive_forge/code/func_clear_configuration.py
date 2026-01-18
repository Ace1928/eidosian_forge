from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def clear_configuration(self):
    configuration = self.get_full_configuration()
    updated = False
    msg = self.NO_CHANGE_MSG
    if configuration['ldapDomains']:
        updated = True
        msg = 'The LDAP configuration for all domains was cleared.'
        if not self.check_mode:
            try:
                rc, result = request(self.url + self.base_path, method='DELETE', ignore_errors=True, **self.creds)
                if rc == 405:
                    for config in configuration['ldapDomains']:
                        self.clear_single_configuration(config['id'])
            except Exception as err:
                self.module.fail_json(msg='Failed to clear LDAP configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
    return (msg, updated)