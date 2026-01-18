from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_node_uuid(self):
    if self._node_uuid is not None:
        return self._node_uuid
    api = 'cluster/nodes'
    query = {'name': self.parameters['node']}
    node, error = rest_generic.get_one_record(self.rest_api, api, query, fields='uuid')
    if error:
        self.module.fail_json(msg='Error reading node UUID: %s' % error)
    if not node:
        self.module.fail_json(msg='Error: node not found %s, current nodes: %s.' % (self.parameters['node'], ', '.join(self.get_node_names())))
    self._node_uuid = node['uuid']
    return node['uuid']