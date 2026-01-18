from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def clear_single_configuration(self, identifier=None):
    if identifier is None:
        identifier = self.identifier
    configuration = self.get_configuration(identifier)
    updated = False
    msg = self.NO_CHANGE_MSG
    if configuration:
        updated = True
        msg = 'The LDAP domain configuration for [%s] was cleared.' % identifier
        if not self.check_mode:
            try:
                rc, result = request(self.url + self.base_path + '%s' % identifier, method='DELETE', **self.creds)
            except Exception as err:
                self.module.fail_json(msg='Failed to remove LDAP configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
    return (msg, updated)