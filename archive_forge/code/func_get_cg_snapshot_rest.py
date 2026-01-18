from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cg_snapshot_rest(self):
    """
        Retrieve CG snapshots using fetched CG uuid
        """
    self.get_cg_rest()
    if self.cg_uuid is None:
        if self.parameters.get('consistency_group') is not None:
            self.module.fail_json(msg="Consistency group named '%s' not found" % self.parameters.get('consistency_group'))
        if self.parameters.get('volumes') is not None:
            self.module.fail_json(msg="Consistency group having volumes '%s' not found" % self.parameters.get('volumes'))
    api = '/application/consistency-groups/%s/snapshots' % self.cg_uuid
    query = {'name': self.parameters['snapshot'], 'fields': 'name,uuid,consistency_group,snapmirror_label,comment,'}
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error searching for consistency group snapshot %s: %s' % (self.parameters['snapshot'], to_native(error)), exception=traceback.format_exc())
    if record:
        return {'snapshot': record.get('name'), 'snapshot_uuid': record.get('uuid'), 'consistency_group': self.na_helper.safe_get(record, ['consistency_group', 'name']), 'snapmirror_label': record.get('snapmirror_label'), 'comment': record.get('comment')}
    return None