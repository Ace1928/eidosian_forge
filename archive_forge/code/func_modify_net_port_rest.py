from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_net_port_rest(self, uuid, modify):
    """
        Modify broadcast domain, ipspace and enable/disable port
        """
    api = 'network/ethernet/ports'
    body = {'enabled': modify['up_admin']}
    dummy, error = rest_generic.patch_async(self.rest_api, api, uuid, body)
    if error:
        self.module.fail_json(msg=error)