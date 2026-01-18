from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def get_node_port(self, uuid):
    record, error = self.get_interface_record_rest(self.parameters['interface_type'], {'uuid': uuid}, 'location')
    if error or not record:
        return (None, None, error)
    node = self.na_helper.safe_get(record, ['location', 'node', 'name'])
    port = self.na_helper.safe_get(record, ['location', 'port', 'name'])
    return (node, port, None)