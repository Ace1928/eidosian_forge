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
def ex_destroy_forwarding_rule(self, forwarding_rule):
    """
        Destroy a forwarding rule.

        :param  forwarding_rule: Forwarding Rule object to destroy
        :type   forwarding_rule: :class:`GCEForwardingRule`

        :return:  True if successful
        :rtype:   ``bool``
        """
    if forwarding_rule.region:
        request = '/regions/{}/forwardingRules/{}'.format(forwarding_rule.region.name, forwarding_rule.name)
    else:
        request = '/global/forwardingRules/%s' % forwarding_rule.name
    self.connection.async_request(request, method='DELETE')
    return True