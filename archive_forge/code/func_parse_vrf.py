from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
def parse_vrf(config):
    match = re.search('vrf (\\w+)', config)
    if match:
        return match.group(1)