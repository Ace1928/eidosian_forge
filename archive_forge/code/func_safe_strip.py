from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def safe_strip(self):
    """ strip the left and right spaces of string and also removes an empty string"""
    for option in ('ciphers', 'key_exchange_algorithms', 'mac_algorithms'):
        if option in self.parameters:
            self.parameters[option] = [item.strip() for item in self.parameters[option] if len(item.strip())]
            if self.parameters[option] == []:
                self.module.fail_json(msg='Removing all SSH %s is not supported. SSH login would fail. There must be at least one %s associated with the SSH configuration.' % (option, option))
    return