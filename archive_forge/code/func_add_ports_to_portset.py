from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def add_ports_to_portset(self, ports_to_add):
    api = 'protocols/san/portsets/%s/interfaces' % self.uuid
    body = {'records': [{self.lifs_info[port]['lif_type']: {'name': port}} for port in ports_to_add]}
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error adding port in portset %s: %s' % (self.parameters['name'], to_native(error)))