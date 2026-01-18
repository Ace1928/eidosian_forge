from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def create_local_host_rest(self):
    """
        Creates a new IP to hostname mapping.
        """
    api = 'name-services/local-hosts'
    body = {'owner.name': self.parameters.get('owner'), 'address': self.parameters.get('address'), 'hostname': self.parameters.get('host')}
    if 'aliases' in self.parameters:
        body['aliases'] = self.parameters.get('aliases')
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating IP to hostname mappings for %s: %s' % (self.parameters['owner'], to_native(error)), exception=traceback.format_exc())