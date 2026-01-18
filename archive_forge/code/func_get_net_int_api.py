from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def get_net_int_api(self, if_type=None):
    if if_type is None:
        if_type = self.parameters.get('interface_type')
    if if_type is None:
        self.module.fail_json(msg='Error: missing option "interface_type (or could not be derived)')
    return 'network/%s/interfaces' % if_type