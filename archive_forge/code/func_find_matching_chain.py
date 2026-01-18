from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.crypto.plugins.module_utils.acme.acme import (
from ansible_collections.community.crypto.plugins.module_utils.acme.account import (
from ansible_collections.community.crypto.plugins.module_utils.acme.challenges import (
from ansible_collections.community.crypto.plugins.module_utils.acme.certificates import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.io import (
from ansible_collections.community.crypto.plugins.module_utils.acme.orders import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
def find_matching_chain(self, chains):
    for criterium_idx, matcher in enumerate(self.select_chain_matcher):
        for chain in chains:
            if matcher.match(chain):
                self.module.debug('Found matching chain for criterium {0}'.format(criterium_idx))
                return chain
    return None