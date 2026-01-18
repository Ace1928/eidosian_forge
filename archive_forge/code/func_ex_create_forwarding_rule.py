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
def ex_create_forwarding_rule(self, name, target=None, region=None, protocol='tcp', port_range=None, address=None, description=None, global_rule=False, targetpool=None, lb_scheme=None):
    """
        Create a forwarding rule.

        :param  name: Name of forwarding rule to be created
        :type   name: ``str``

        :keyword  target: The target of this forwarding rule.  For global
                          forwarding rules this must be a global
                          TargetHttpProxy. For regional rules this may be
                          either a TargetPool or TargetInstance. If passed
                          a string instead of the object, it will be the name
                          of a TargetHttpProxy for global rules or a
                          TargetPool for regional rules.  A TargetInstance
                          must be passed by object. (required)
        :type     target: ``str`` or :class:`GCETargetHttpProxy` or
                          :class:`GCETargetInstance` or :class:`GCETargetPool`

        :keyword  region: Region to create the forwarding rule in.  Defaults to
                          self.region.  Ignored if global_rule is True.
        :type     region: ``str`` or :class:`GCERegion`

        :keyword  protocol: Should be 'tcp' or 'udp'
        :type     protocol: ``str``

        :keyword  port_range: Single port number or range separated by a dash.
                              Examples: '80', '5000-5999'.  Required for global
                              forwarding rules, optional for regional rules.
        :type     port_range: ``str``

        :keyword  address: Optional static address for forwarding rule. Must be
                           in same region.
        :type     address: ``str`` or :class:`GCEAddress`

        :keyword  description: The description of the forwarding rule.
                               Defaults to None.
        :type     description: ``str`` or ``None``

        :keyword  targetpool: Deprecated parameter for backwards compatibility.
                              Use target instead.
        :type     targetpool: ``str`` or :class:`GCETargetPool`

        :keyword  lb_scheme: Load balancing scheme, can be 'EXTERNAL' or
                             'INTERNAL'. Defaults to 'EXTERNAL'.
        :type     lb_scheme: ``str`` or ``None``

        :return:  Forwarding Rule object
        :rtype:   :class:`GCEForwardingRule`
        """
    forwarding_rule_data = {'name': name}
    if global_rule:
        if not hasattr(target, 'name'):
            target = self.ex_get_targethttpproxy(target)
    else:
        region = region or self.region
        if not hasattr(region, 'name'):
            region = self.ex_get_region(region)
        forwarding_rule_data['region'] = region.extra['selfLink']
        if not target:
            target = targetpool
        if not hasattr(target, 'name'):
            target = self.ex_get_targetpool(target, region)
    forwarding_rule_data['target'] = target.extra['selfLink']
    forwarding_rule_data['IPProtocol'] = protocol.upper()
    if address:
        if not hasattr(address, 'name'):
            address = self.ex_get_address(address, 'global' if global_rule else region)
        forwarding_rule_data['IPAddress'] = address.address
    if port_range:
        forwarding_rule_data['portRange'] = port_range
    if description:
        forwarding_rule_data['description'] = description
    if lb_scheme:
        forwarding_rule_data['loadBalancingScheme'] = lb_scheme
    if global_rule:
        request = '/global/forwardingRules'
    else:
        request = '/regions/%s/forwardingRules' % region.name
    self.connection.async_request(request, method='POST', data=forwarding_rule_data)
    return self.ex_get_forwarding_rule(name, global_rule=global_rule)