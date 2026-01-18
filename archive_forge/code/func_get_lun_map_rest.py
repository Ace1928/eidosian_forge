from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
import codecs
from ansible.module_utils._text import to_text, to_bytes
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_lun_map_rest(self):
    api = 'protocols/san/lun-maps'
    params = {'lun.name': self.parameters['path'], 'svm.name': self.parameters['vserver'], 'igroup.name': self.parameters['initiator_group_name'], 'fields': 'logical_unit_number,igroup.uuid,lun.uuid,lun.name,igroup.name'}
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error getting lun_map %s: %s' % (self.parameters['path'], error))
    if record:
        return {'lun_id': str(self.na_helper.safe_get(record, ['logical_unit_number'])), 'igroup_uuid': self.na_helper.safe_get(record, ['igroup', 'uuid']), 'initiator_group_name': self.na_helper.safe_get(record, ['igroup', 'name']), 'lun_uuid': self.na_helper.safe_get(record, ['lun', 'uuid']), 'path': self.na_helper.safe_get(record, ['lun', 'name'])}
    return None