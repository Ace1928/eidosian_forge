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
def is_first_step(self):
    """
        Return True if this is the first execution of this module, i.e. if a
        sufficient data object from a first run has not been provided.
        """
    if self.data is None:
        return True
    if self.version == 1:
        return not self.data
    else:
        return self.order_uri is None