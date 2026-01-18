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
def ex_get_firewall(self, name):
    """
        Return a Firewall object based on the firewall name.

        :param  name: The name of the firewall
        :type   name: ``str``

        :return:  A GCEFirewall object
        :rtype:   :class:`GCEFirewall`
        """
    request = '/global/firewalls/%s' % name
    response = self.connection.request(request, method='GET').object
    return self._to_firewall(response)