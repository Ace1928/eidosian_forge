from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def migrate_interface_rest(self, uuid, body):
    errors = []
    desired_node = self.na_helper.safe_get(body, ['location', 'node', 'name'])
    desired_port = self.na_helper.safe_get(body, ['location', 'port', 'name'])
    for __ in range(12):
        self.modify_interface_rest(uuid, body)
        time.sleep(10)
        node, port, error = self.get_node_port(uuid)
        if error is None and desired_node in [None, node] and (desired_port in [None, port]):
            return
        if errors or error is not None:
            errors.append(str(error))
    if errors:
        self.module.fail_json(msg='Errors waiting for migration to complete: %s' % ' - '.join(errors))
    else:
        self.module.warn('Failed to confirm interface is migrated after 120 seconds')