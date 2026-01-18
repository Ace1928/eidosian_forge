from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def add_user_to_pool_owners(user, pool):
    """
    Find the current list of pool owners and add user using pool.set_owners().
    set_owners() replaces the current owners with the list of new owners. So,
    get owners, add user, then set owners.  Further, we need to know if the
    owners changed.  Use sets of owners to compare.
    """
    changed = False
    pool_fields = pool.get_fields(from_cache=True, raw_value=True)
    pool_owners = pool_fields.get('owners', [])
    pool_owners_set = set(pool_owners)
    new_pool_owners_set = pool_owners_set.copy()
    new_pool_owners_set.add(user.id)
    if pool_owners_set != new_pool_owners_set:
        pool.set_owners([user])
        changed = True
    return changed