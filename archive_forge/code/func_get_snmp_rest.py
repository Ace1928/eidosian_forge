from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_snmp_rest(self):
    api = 'support/snmp/users'
    params = {'name': self.parameters['snmp_username'], 'fields': 'name,engine_id'}
    message, error = self.rest_api.get(api, params)
    record, error = rrh.check_for_0_or_1_records(api, message, error)
    if error:
        self.module.fail_json(msg=error)
    if record:
        return dict(snmp_username=record['name'], engine_id=record['engine_id'], access_control='ro')
    return None