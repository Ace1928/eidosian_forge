from __future__ import (absolute_import, division, print_function)
import re
from ansible_collections.community.routeros.plugins.module_utils.routeros import run_commands
from ansible_collections.community.routeros.plugins.module_utils.routeros import routeros_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_instance_name(self, data):
    match = re.search('instance=([\\w\\d\\-]+)', data, re.M)
    if match:
        return match.group(1)