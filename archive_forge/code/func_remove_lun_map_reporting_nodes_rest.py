from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def remove_lun_map_reporting_nodes_rest(self, node):
    api = 'protocols/san/lun-maps/%s/%s/reporting-nodes' % (self.lun_uuid, self.igroup_uuid)
    dummy, error = rest_generic.delete_async(self.rest_api, api, self.nodes_uuids[node])
    if error:
        self.module.fail_json(msg='Error deleting LUN map reporting nodes for %s: %s' % (self.parameters['initiator_group_name'], to_native(error)))