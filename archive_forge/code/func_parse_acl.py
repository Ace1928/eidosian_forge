from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_acl(line, dest):
    if dest == 'access-group':
        match = re.search('ntp\\saccess-group\\s(?:peer|serve)(?:\\s+)(\\S+)', line, re.M)
        if match:
            acl = match.group(1)
            return acl