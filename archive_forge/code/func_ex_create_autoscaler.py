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
def ex_create_autoscaler(self, name, zone, instance_group, policy, description=None):
    """
        Create an Autoscaler for an Instance Group.

        :param  name: The name of the Autoscaler
        :type   name: ``str``

        :param  zone: The zone to which the Instance Group belongs
        :type   zone: ``str`` or :class:`GCEZone`

        :param  instance_group:  An Instance Group Manager object.
        :type:  :class:`GCEInstanceGroupManager`

        :param  policy:  A dict containing policy configuration.  See the
                         API documentation for Autoscalers for more details.
        :type:  ``dict``

        :return:  An Autoscaler object.
        :rtype:   :class:`GCEAutoscaler`
        """
    zone = zone or self.zone
    autoscaler_data = {}
    autoscaler_data = {'name': name}
    if not hasattr(zone, 'name'):
        zone = self.ex_get_zone(zone)
    autoscaler_data['zone'] = zone.extra['selfLink']
    autoscaler_data['autoscalingPolicy'] = policy
    request = '/zones/%s/autoscalers' % zone.name
    autoscaler_data['target'] = instance_group.extra['selfLink']
    self.connection.async_request(request, method='POST', data=autoscaler_data)
    return self.ex_get_autoscaler(name, zone)