from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_adapter_uuid(self, adapter):
    api = 'network/fc/ports'
    params = {'name': adapter, 'node.name': self.parameters['node_name'], 'fields': 'uuid'}
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error fetching adapter %s uuid' % adapter)
    return record['uuid'] if record else None