import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_create_targetinstance(self, name, zone=None, node=None, description=None, nat_policy='NO_NAT'):
    """
        Create a target instance.

        :param  name: Name of target instance
        :type   name: ``str``

        :keyword  region: Zone to create the target pool in. Defaults to
                          self.zone
        :type     region: ``str`` or :class:`GCEZone` or ``None``

        :keyword  node: The actual instance to be used as the traffic target.
        :type     node: ``str`` or :class:`Node`

        :keyword  description: A text description for the target instance
        :type     description: ``str`` or ``None``

        :keyword  nat_policy: The NAT option for how IPs are NAT'd to the node.
        :type     nat_policy: ``str``

        :return:  Target Instance object
        :rtype:   :class:`GCETargetInstance`
        """
    zone = zone or self.zone
    targetinstance_data = {}
    targetinstance_data['name'] = name
    if not hasattr(zone, 'name'):
        zone = self.ex_get_zone(zone)
    targetinstance_data['zone'] = zone.extra['selfLink']
    if node is not None:
        if not hasattr(node, 'name'):
            node = self.ex_get_node(node, zone)
        targetinstance_data['instance'] = node.extra['selfLink']
    targetinstance_data['natPolicy'] = nat_policy
    if description:
        targetinstance_data['description'] = description
    request = '/zones/%s/targetInstances' % zone.name
    self.connection.async_request(request, method='POST', data=targetinstance_data)
    return self.ex_get_targetinstance(name, zone)