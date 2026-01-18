from __future__ import (absolute_import, division, print_function)
from ansible.utils.vars import combine_vars
def get_group_vars(groups):
    """
    Combine all the group vars from a list of inventory groups.

    :param groups: list of ansible.inventory.group.Group objects
    :rtype: dict
    """
    results = {}
    for group in sort_groups(groups):
        results = combine_vars(results, group.get_vars())
    return results