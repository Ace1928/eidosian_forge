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
def ex_create_subnetwork(self, name, cidr=None, network=None, region=None, description=None, privateipgoogleaccess=None, secondaryipranges=None):
    """
        Create a subnetwork.

        :param  name: Name of subnetwork to be created
        :type   name: ``str``

        :param  cidr: Address range of network in CIDR format.
        :type   cidr: ``str``

        :param  network: The network name or object this subnet belongs to.
        :type   network: ``str`` or :class:`GCENetwork`

        :param  region: The region the subnetwork belongs to.
        :type   region: ``str`` or :class:`GCERegion`

        :param  description: Custom description for the network.
        :type   description: ``str`` or ``None``

        :param  privateipgoogleaccess: Allow access to Google services without
                                       assigned external IP addresses.
        :type   privateipgoogleaccess: ``bool` or ``None``

        :param  secondaryipranges: List of dicts of secondary or "alias" IP
                                   ranges for this subnetwork in
                                   [{"rangeName": "second1",
                                   "ipCidrRange": "192.168.168.0/24"},
                                   {k:v, k:v}] format.
        :type   secondaryipranges: ``list`` of ``dict`` or ``None``

        :return:  Subnetwork object
        :rtype:   :class:`GCESubnetwork`
        """
    if not cidr:
        raise ValueError('Must provide an IP network in CIDR notation.')
    if not network:
        raise ValueError('Must provide a network for the subnetwork.')
    elif isinstance(network, GCENetwork):
        network_url = network.extra['selfLink']
    elif network.startswith('https://'):
        network_url = network
    else:
        network_obj = self.ex_get_network(network)
        network_url = network_obj.extra['selfLink']
    if not region:
        raise ValueError('Must provide a region for the subnetwork.')
    elif isinstance(region, GCERegion):
        region_url = region.extra['selfLink']
    elif region.startswith('https://'):
        region_url = region
    else:
        region_obj = self.ex_get_region(region)
        region_url = region_obj.extra['selfLink']
    subnet_data = {}
    subnet_data['name'] = name
    subnet_data['description'] = description
    subnet_data['ipCidrRange'] = cidr
    subnet_data['network'] = network_url
    subnet_data['region'] = region_url
    subnet_data['privateIpGoogleAccess'] = privateipgoogleaccess
    subnet_data['secondaryIpRanges'] = secondaryipranges
    region_name = region_url.split('/')[-1]
    request = '/regions/%s/subnetworks' % region_name
    self.connection.async_request(request, method='POST', data=subnet_data)
    return self.ex_get_subnetwork(name, region_name)