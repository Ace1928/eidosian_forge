from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def check_existing(self, name, query):
    """Helper method to lookup existing states on an interface.
        This is needed for attribute changes that have additional dependencies;
        e.g. 'ip redirects' may auto-enable when all secondary ip addrs are removed.
        """
    if name:
        have = self.existing_facts.get(name, {})
        if 'has_secondary' in query:
            return have.get('has_secondary', False)
        if 'redirects' in query:
            return have.get('redirects', True)
        if 'unreachables' in query:
            return have.get('unreachables', False)
    return None