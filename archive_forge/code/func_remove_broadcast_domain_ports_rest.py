from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def remove_broadcast_domain_ports_rest(self, ports, ipspace):
    body = {'ports': ports}
    api = 'private/cli/network/port/broadcast-domain/remove-ports'
    query = {'broadcast-domain': self.parameters['resource_name'], 'ipspace': ipspace}
    response, error = rest_generic.patch_async(self.rest_api, api, None, body, query)
    if error:
        self.module.fail_json(msg='Error removing ports: %s' % error)