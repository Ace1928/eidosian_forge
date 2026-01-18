from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import netapp_ipaddress
def change_status_to_bool(self, input, to_zapi=True):
    if to_zapi:
        return 'true' if input == 'enable' else 'false'
    else:
        return 'enable' if input == 'true' else 'disable'