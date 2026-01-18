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
def ex_get_instancegroup(self, name, zone=None):
    """
        Returns the specified Instance Group. Get a list of available instance
        groups by making a list() request.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute
        * https://www.googleapis.com/auth/compute.readonly

        :param  name:  The name of the instance group.
        :type   name: ``str``

        :param  zone:  The name of the zone where the instance group is
                       located.
        :type   zone: ``str``

        :return:  `GCEInstanceGroup` object.
        :rtype:   :class:`GCEInstanceGroup`
        """
    zone = self._set_zone(zone) or self._find_zone_or_region(name, 'instanceGroups', region=False, res_name='Instancegroup')
    request = '/zones/{}/instanceGroups/{}'.format(zone.name, name)
    response = self.connection.request(request, method='GET').object
    return self._to_instancegroup(response)