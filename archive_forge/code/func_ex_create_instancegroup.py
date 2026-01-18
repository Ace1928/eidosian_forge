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
def ex_create_instancegroup(self, name, zone, description=None, network=None, subnetwork=None, named_ports=None):
    """
        Creates an instance group in the specified project using the
        parameters that are included in the request.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  name:  Required. The name of the instance group. The name
                       must be 1-63 characters long, and comply with RFC1035.
        :type   name: ``str``

        :param  zone:  The URL of the zone where the instance group is
                       located.
        :type   zone: :class:`GCEZone`

        :keyword  description:  An optional description of this resource.
                                Provide this property when you create the
                                resource.
        :type   description: ``str``

        :keyword  network:  The URL of the network to which all instances in
                            the instance group belong.
        :type   network: :class:`GCENetwork`

        :keyword  subnetwork:  The URL of the subnetwork to which all
                               instances in the instance group belong.
        :type   subnetwork: :class:`GCESubnetwork`

        :keyword  named_ports:  Assigns a name to a port number. For example:
                                {name: "http", port: 80}  This allows the
                                system to reference ports by the assigned
                                name instead of a port number. Named ports
                                can also contain multiple ports. For example:
                                [{name: "http", port: 80},{name: "http",
                                port: 8080}]   Named ports apply to all
                                instances in this instance group.
        :type   named_ports: ``list`` of {'name': ``str``, 'port`: ``int``}

        :return:  `GCEInstanceGroup` object.
        :rtype: :class:`GCEInstanceGroup`
        """
    zone = zone or self.zone
    if not hasattr(zone, 'name'):
        zone = self.ex_get_zone(zone)
    request = '/zones/%s/instanceGroups' % zone.name
    request_data = {}
    request_data['name'] = name
    request_data['zone'] = zone.extra['selfLink']
    if description:
        request_data['description'] = description
    if network:
        request_data['network'] = network.extra['selfLink']
    if subnetwork:
        request_data['subnetwork'] = subnetwork.extra['selfLink']
    if named_ports:
        request_data['namedPorts'] = named_ports
    self.connection.async_request(request, method='POST', data=request_data)
    return self.ex_get_instancegroup(name, zone)