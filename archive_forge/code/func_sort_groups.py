from __future__ import (absolute_import, division, print_function)
from ansible.utils.vars import combine_vars
def sort_groups(groups):
    return sorted(groups, key=lambda g: (g.depth, g.priority, g.name))