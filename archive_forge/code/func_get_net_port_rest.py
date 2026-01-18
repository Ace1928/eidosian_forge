from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_net_port_rest(self, port):
    if ':' not in port:
        error_msg = 'Error: Invalid value specified for port: %s, provide port name as node_name:port_name' % port
        self.module.fail_json(msg=error_msg)
    node_name, port_name = port.split(':')
    api = 'network/ethernet/ports'
    query = {'name': port_name, 'node.name': node_name}
    fields = 'name,uuid'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg=error)
    if record:
        current = {'uuid': record['uuid'], 'name': '%s:%s' % (record['node']['name'], record['name'])}
        return current
    return None