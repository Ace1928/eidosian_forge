from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def build_rest_body(self, modify=None):
    required_keys = set(['interface_type'])
    self.validate_required_parameters(required_keys)
    self.validate_rest_input_parameters(action='modify' if modify else 'create')
    if modify:
        if self.parameters.get('fail_if_subnet_conflicts') is not None:
            modify['fail_if_subnet_conflicts'] = self.parameters['fail_if_subnet_conflicts']
    else:
        required_keys = set()
        required_keys.add('interface_name')
        if self.parameters['interface_type'] == 'fc':
            self.derive_fc_data_protocol()
            required_keys.add('data_protocol')
            if 'home_port' not in self.parameters:
                if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 8, 0):
                    required_keys.add('home_port')
                else:
                    required_keys.add('current_port')
        if self.parameters['interface_type'] == 'ip':
            if 'subnet_name' not in self.parameters:
                required_keys.add('address')
                required_keys.add('netmask')
            required_keys.add('broadcast_domain_home_port_or_home_node')
        self.validate_required_parameters(required_keys)
    body, migrate_body, errors = self.set_options_rest(modify)
    self.fix_errors(body, errors)
    if errors:
        self.module.fail_json(msg='Error %s interface, unsupported options: %s' % ('modifying' if modify else 'creating', str(errors)))
    if modify:
        self.validate_modify_parameters(body)
    return (body, migrate_body)