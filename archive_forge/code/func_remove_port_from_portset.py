from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def remove_port_from_portset(self, port_to_remove):
    api = 'protocols/san/portsets/%s/interfaces' % self.uuid
    dummy, error = rest_generic.delete_async(self.rest_api, api, self.lifs_info[port_to_remove]['uuid'])
    if error:
        self.module.fail_json(msg='Error removing port in portset %s: %s' % (self.parameters['name'], to_native(error)))