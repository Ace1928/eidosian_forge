from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def modify_local_host_rest(self, modify):
    """
        For a specified SVM and IP address, modifies the corresponding IP to hostname mapping.
        """
    body = {}
    if 'aliases' in modify:
        body['aliases'] = self.parameters['aliases']
    if 'host' in modify:
        body['hostname'] = self.parameters['host']
    api = 'name-services/local-hosts/%s/%s' % (self.owner_uuid, self.parameters['address'])
    if body:
        dummy, error = rest_generic.patch_async(self.rest_api, api, None, body)
        if error:
            self.module.fail_json(msg='Error updating IP to hostname mappings for %s: %s' % (self.parameters['owner'], to_native(error)), exception=traceback.format_exc())