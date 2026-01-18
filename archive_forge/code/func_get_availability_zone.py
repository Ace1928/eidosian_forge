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
def get_availability_zone(self, *attrs):
    """Get an availability zone

        :param availability_zone: The value can be the ID of a
            availability_zone or
            :class:`~openstack.load_balancer.v2.availability_zone.AvailabilityZone`
            instance.

        :returns: One
            :class:`~openstack.load_balancer.v2.availability_zone.AvailabilityZone`
        """
    return self._get(_availability_zone.AvailabilityZone, *attrs)