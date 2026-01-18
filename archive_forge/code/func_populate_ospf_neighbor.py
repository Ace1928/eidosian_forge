from __future__ import (absolute_import, division, print_function)
import re
from ansible_collections.community.routeros.plugins.module_utils.routeros import run_commands
from ansible_collections.community.routeros.plugins.module_utils.routeros import routeros_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def populate_ospf_neighbor(self, data):
    for key, value in iteritems(data):
        self.facts['ospf_neighbor'][key] = value