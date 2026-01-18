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
def delete_l7_rule(self, l7rule, l7_policy, ignore_missing=True):
    """Delete a l7rule

        :param l7rule: The l7rule can be either the ID of a l7rule or a
            :class:`~openstack.load_balancer.v2.l7_rule.L7Rule` instance.
        :param l7_policy: The l7_policy can be either the ID of a l7policy or
            :class:`~openstack.load_balancer.v2.l7_policy.L7Policy`
            instance that the l7rule belongs to.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the l7rule does not exist.
            When set to ``True``, no exception will be set when
            attempting to delete a nonexistent l7rule.

        :returns: ``None``
        """
    l7policyobj = self._get_resource(_l7policy.L7Policy, l7_policy)
    self._delete(_l7rule.L7Rule, l7rule, ignore_missing=ignore_missing, l7policy_id=l7policyobj.id)