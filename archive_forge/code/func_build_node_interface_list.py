from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def build_node_interface_list(self):
    node_interfaces = []
    api = 'api/v3/grid/node-health'
    nodes, error = self.rest_api.get(api)
    if error:
        self.module.fail_json(msg=error)
    for node_interface in self.parameters['node_interfaces']:
        node_dict = {}
        node = next((item for item in nodes['data'] if item['name'] == node_interface['node']), None)
        if node is not None:
            node_dict['nodeId'] = node['id']
            node_dict['interface'] = node_interface['interface']
            node_interfaces.append(node_dict)
        else:
            self.module.fail_json(msg="Node '%s' is invalid" % node_interface['node'])
    return node_interfaces