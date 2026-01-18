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
def ex_get_accelerator_type(self, name, zone=None):
    """
        Return an AcceleratorType object based on a name and zone.

        :param  name: The name of the AcceleratorType
        :type   name: ``str``

        :param  zone: The zone to search for the AcceleratorType in.
        :type   zone: :class:`GCEZone`

        :return:  An AcceleratorType object for the name
        :rtype:   :class:`GCEAcceleratorType`
        """
    zone = self._set_zone(zone)
    request = '/zones/{}/acceleratorTypes/{}'.format(zone.name, name)
    response = self.connection.request(request, method='GET').object
    return self._to_accelerator_type(response)