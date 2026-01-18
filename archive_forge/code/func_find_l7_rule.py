from openstack.load_balancer.v2 import amphora as _amphora
from openstack.load_balancer.v2 import availability_zone as _availability_zone
from openstack.load_balancer.v2 import (
from openstack.load_balancer.v2 import flavor as _flavor
from openstack.load_balancer.v2 import flavor_profile as _flavor_profile
from openstack.load_balancer.v2 import health_monitor as _hm
from openstack.load_balancer.v2 import l7_policy as _l7policy
from openstack.load_balancer.v2 import l7_rule as _l7rule
from openstack.load_balancer.v2 import listener as _listener
from openstack.load_balancer.v2 import load_balancer as _lb
from openstack.load_balancer.v2 import member as _member
from openstack.load_balancer.v2 import pool as _pool
from openstack.load_balancer.v2 import provider as _provider
from openstack.load_balancer.v2 import quota as _quota
from openstack import proxy
from openstack import resource
def find_l7_rule(self, name_or_id, l7_policy, ignore_missing=True):
    """Find a single l7rule

        :param str name_or_id: The name or ID of a l7rule.
        :param l7_policy: The l7_policy can be either the ID of a l7policy or
            :class:`~openstack.load_balancer.v2.l7_policy.L7Policy`
            instance that the l7rule belongs to.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the resource does not exist.
            When set to ``True``, None will be returned when
            attempting to find a nonexistent resource.

        :returns: One :class:`~openstack.load_balancer.v2.l7_rule.L7Rule`
            or None
        """
    l7policyobj = self._get_resource(_l7policy.L7Policy, l7_policy)
    return self._find(_l7rule.L7Rule, name_or_id, ignore_missing=ignore_missing, l7policy_id=l7policyobj.id)