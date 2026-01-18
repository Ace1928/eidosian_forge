import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_create_vpc(self, cidr, display_text, name, vpc_offering, zone_id, network_domain=None):
    """

        Creates a VPC, only available in advanced zones.

        :param  cidr: the cidr of the VPC. All VPC guest networks' cidrs
                should be within this CIDR

        :type   display_text: ``str``

        :param  display_text: the display text of the VPC
        :type   display_text: ``str``

        :param  name: the name of the VPC
        :type   name: ``str``

        :param  vpc_offering: the ID of the VPC offering
        :type   vpc_offering: :class:'CloudStackVPCOffering`

        :param  zone_id: the ID of the availability zone
        :type   zone_id: ``str``

        :param  network_domain: Optional, the DNS domain of the network
        :type   network_domain: ``str``

        :rtype: :class:`CloudStackVPC`

        """
    extra_map = RESOURCE_EXTRA_ATTRIBUTES_MAP['vpc']
    args = {'cidr': cidr, 'displaytext': display_text, 'name': name, 'vpcofferingid': vpc_offering.id, 'zoneid': zone_id}
    if network_domain is not None:
        args['networkdomain'] = network_domain
    result = self._sync_request(command='createVPC', params=args, method='GET')
    extra = self._get_extra_dict(result, extra_map)
    vpc = CloudStackVPC(name, vpc_offering.id, result['id'], cidr, self, zone_id, display_text, extra=extra)
    return vpc