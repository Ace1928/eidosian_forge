from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_cg_snapshot_rest(self, current):
    """Delete CG snapshot"""
    api = '/application/consistency-groups/%s/snapshots' % self.cg_uuid
    dummy, error = rest_generic.delete_async(self.rest_api, api, current['snapshot_uuid'])
    if error:
        self.module.fail_json(msg='Error deleting consistency group snapshot %s: %s' % (self.parameters['snapshot'], to_native(error)), exception=traceback.format_exc())