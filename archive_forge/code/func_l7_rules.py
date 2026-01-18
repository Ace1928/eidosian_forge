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
def l7_rules(self, l7_policy, **query):
    """Return a generator of l7rules

        :param l7_policy: The l7_policy can be either the ID of a l7_policy or
            :class:`~openstack.load_balancer.v2.l7_policy.L7Policy`
            instance that the l7rule belongs to.
        :param dict query: Optional query parameters to be sent to limit
            the resources being returned. Valid parameters are:

        :returns: A generator of l7rule objects
        :rtype: :class:`~openstack.load_balancer.v2.l7_rule.L7Rule`
        """
    l7policyobj = self._get_resource(_l7policy.L7Policy, l7_policy)
    return self._list(_l7rule.L7Rule, l7policy_id=l7policyobj.id, **query)