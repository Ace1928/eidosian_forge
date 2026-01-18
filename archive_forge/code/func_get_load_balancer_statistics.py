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
def get_load_balancer_statistics(self, load_balancer):
    """Get the load balancer statistics

        :param load_balancer: The value can be the ID of a load balancer
            or :class:`~openstack.load_balancer.v2.load_balancer.LoadBalancer`
            instance.

        :returns: One
            :class:`~openstack.load_balancer.v2.load_balancer.LoadBalancerStats`
        """
    return self._get(_lb.LoadBalancerStats, lb_id=load_balancer, requires_id=False)