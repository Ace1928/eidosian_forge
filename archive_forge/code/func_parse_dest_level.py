from __future__ import absolute_import, division, print_function
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def parse_dest_level(line, dest, name):
    dest_level = None

    def parse_match(match):
        level = None
        if match:
            if int(match.group(1)) in range(0, 8):
                level = match.group(1)
            else:
                pass
        return level
    if dest and dest != 'server':
        if dest == 'logfile':
            match = re.search('logging logfile {0} (\\S+)'.format(name), line, re.M)
            if match:
                dest_level = parse_match(match)
        elif dest == 'server':
            match = re.search('logging server (?:\\S+) (\\d+)', line, re.M)
            if match:
                dest_level = parse_match(match)
        else:
            match = re.search('logging {0} (\\S+)'.format(dest), line, re.M)
            if match:
                dest_level = parse_match(match)
    return dest_level