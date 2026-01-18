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
def ex_list_forwarding_rules(self, region=None, global_rules=False):
    """
        Return the list of forwarding rules for a region or all.

        :keyword  region: The region to return forwarding rules from.  For
                          example: 'us-central1'.  If None, will return
                          forwarding rules from the region of self.region
                          (which is based on self.zone).  If 'all', will
                          return forwarding rules for all regions, which does
                          not include the global forwarding rules.
        :type     region: ``str`` or :class:`GCERegion` or ``None``

        :keyword  global_rules: List global forwarding rules instead of
                                per-region rules.  Setting True will cause
                                'region' parameter to be ignored.
        :type     global_rules: ``bool``

        :return: A list of forwarding rule objects.
        :rtype: ``list`` of :class:`GCEForwardingRule`
        """
    list_forwarding_rules = []
    if global_rules:
        region = None
        request = '/global/forwardingRules'
    else:
        region = self._set_region(region)
        if region is None:
            request = '/aggregated/forwardingRules'
        else:
            request = '/regions/%s/forwardingRules' % region.name
    response = self.connection.request(request, method='GET').object
    if 'items' in response:
        if not global_rules and region is None:
            for v in response['items'].values():
                region_forwarding_rules = [self._to_forwarding_rule(f) for f in v.get('forwardingRules', [])]
                list_forwarding_rules.extend(region_forwarding_rules)
        else:
            list_forwarding_rules = [self._to_forwarding_rule(f) for f in response['items']]
    return list_forwarding_rules