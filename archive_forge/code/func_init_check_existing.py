from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def init_check_existing(self, have):
    """Creates a class var dict for easier access to existing states"""
    self.existing_facts = dict()
    have_copy = deepcopy(have)
    for intf in have_copy:
        name = intf['name']
        self.existing_facts[name] = intf
        if [i for i in intf.get('ipv4', []) if i.get('secondary')]:
            self.existing_facts[name]['has_secondary'] = True