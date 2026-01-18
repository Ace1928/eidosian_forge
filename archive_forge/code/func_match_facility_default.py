from __future__ import absolute_import, division, print_function
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def match_facility_default(module, facility, want_level):
    """Check wanted facility to see if it matches current device default"""
    matches_default = False
    regexl = '\\S+\\s+(\\d+)\\s+(\\d+)'
    cmd = {'command': 'show logging level {0}'.format(facility), 'output': 'text'}
    facility_data = run_commands(module, cmd)
    for line in facility_data[0].split('\n'):
        mo = re.search(regexl, line)
        if mo and int(mo.group(1)) == int(want_level) and (int(mo.group(2)) == int(want_level)):
            matches_default = True
    return matches_default