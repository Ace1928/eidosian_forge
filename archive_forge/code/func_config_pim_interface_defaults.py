from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def config_pim_interface_defaults(existing, jp_bidir, isauth):
    command = []
    defaults = get_pim_interface_defaults()
    delta = dict(set(defaults.items()).difference(existing.items()))
    if delta:
        command = config_pim_interface(delta, existing, jp_bidir, isauth)
    comm = default_pim_interface_policies(existing, jp_bidir)
    if comm:
        for each in comm:
            command.append(each)
    return command