from __future__ import (absolute_import, division, print_function)
from collections.abc import MutableMapping
from ansible.utils.vars import merge_hash
def set_custom_stats(self, which, what, host=None):
    """ allow setting of a custom stat"""
    if host is None:
        host = '_run'
    if host not in self.custom:
        self.custom[host] = {which: what}
    else:
        self.custom[host][which] = what