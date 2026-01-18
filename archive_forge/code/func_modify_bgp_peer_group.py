from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def modify_bgp_peer_group(self, modify):
    """
        Modify BGP peer group.
        """
    api = 'network/ip/bgp/peer-groups'
    body = {}
    if 'name' in modify:
        body['name'] = modify['name']
    if 'peer' in modify:
        body['peer'] = modify['peer']
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body)
    if error:
        name = self.parameters['from_name'] if 'name' in modify else self.parameters['name']
        self.module.fail_json(msg='Error modifying BGP peer group %s: %s.' % (name, to_native(error)), exception=traceback.format_exc())