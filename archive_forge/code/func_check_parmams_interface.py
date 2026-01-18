from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import load_config, run_commands
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import build_aggregate_spec, map_params_to_obj
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch_interface import InterfaceConfiguration, merge_interfaces
def check_parmams_interface(interfaces):
    if interfaces:
        for i in interfaces:
            match = re.search('(\\d+)\\/(\\d+)-(\\d+)\\/(\\d+)', i)
            if match:
                if match.group(1) != match.group(3):
                    module.fail_json(msg='interface range must be within same group: ' + i)
            else:
                match = re.search('(\\d+)\\/(\\d+)', i)
                if not match:
                    module.fail_json(msg='wrong interface format: ' + i)