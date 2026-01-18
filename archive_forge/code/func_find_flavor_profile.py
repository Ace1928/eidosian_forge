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
def find_flavor_profile(self, name_or_id, ignore_missing=True):
    """Find a single flavor profile

        :param name_or_id: The name or ID of a flavor profile
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised
            when the flavor profile does not exist.
            When set to ``True``, no exception will be set when attempting
            to delete a nonexistent flavor profile.

        :returns: ``None``
        """
    return self._find(_flavor_profile.FlavorProfile, name_or_id, ignore_missing=ignore_missing)