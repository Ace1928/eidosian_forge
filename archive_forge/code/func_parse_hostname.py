from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_hostname(config):
    match = re.search('^hostname (\\S+)', config, re.M)
    return match.group(1)