from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def parse_logging(line, dest):
    if dest == 'logging':
        logging = dest
        return logging