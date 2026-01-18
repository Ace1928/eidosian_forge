from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_sp_state(self):
    api = 'cluster/nodes/%s' % self.get_node_uuid()
    node, error = rest_generic.get_one_record(self.rest_api, api, fields='service_processor.state')
    if error:
        self.module.fail_json(msg='Error getting node SP state: %s' % error)
    if node:
        return self.na_helper.safe_get(node, ['service_processor', 'state'])