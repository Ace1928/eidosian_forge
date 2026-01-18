import copy
import itertools
from oslo_log import log
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def add_implied_roles(self, role_refs):
    """Expand out implied roles.

        The role_refs passed in have had all inheritance and group assignments
        expanded out. We now need to look at the role_id in each ref and see
        if it is a prior role for some implied roles. If it is, then we need to
        duplicate that ref, one for each implied role. We store the prior role
        in the indirect dict that is part of such a duplicated ref, so that a
        caller can determine where the assignment came from.

        """

    def _make_implied_ref_copy(prior_ref, implied_role_id):
        implied_ref = copy.deepcopy(prior_ref)
        implied_ref['role_id'] = implied_role_id
        indirect = implied_ref.setdefault('indirect', {})
        indirect['role_id'] = prior_ref['role_id']
        return implied_ref
    try:
        implied_roles_cache = {}
        role_refs_to_check = list(role_refs)
        ref_results = list(role_refs)
        checked_role_refs = list()
        while role_refs_to_check:
            next_ref = role_refs_to_check.pop()
            checked_role_refs.append(next_ref)
            next_role_id = next_ref['role_id']
            if next_role_id in implied_roles_cache:
                implied_roles = implied_roles_cache[next_role_id]
            else:
                implied_roles = PROVIDERS.role_api.list_implied_roles(next_role_id)
                implied_roles_cache[next_role_id] = implied_roles
            for implied_role in implied_roles:
                implied_ref = _make_implied_ref_copy(next_ref, implied_role['implied_role_id'])
                if implied_ref in checked_role_refs:
                    continue
                else:
                    ref_results.append(implied_ref)
                    role_refs_to_check.append(implied_ref)
    except exception.NotImplemented:
        LOG.error('Role driver does not support implied roles.')
    return ref_results