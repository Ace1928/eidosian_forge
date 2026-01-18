from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import get_config, load_config
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import cnos_argument_spec
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import debugOutput, run_commands
from ansible.module_utils.connection import exec_command
def get_ethernet_range(module):
    output = run_commands(module, ['show interface brief'])[0].split('\n')
    maxport = None
    last_interface = None
    for line in output:
        if line.startswith('Ethernet1/'):
            last_interface = line.split(' ')[0]
    if last_interface is not None:
        eths = last_interface.split('/')
        maxport = eths[1]
    return maxport