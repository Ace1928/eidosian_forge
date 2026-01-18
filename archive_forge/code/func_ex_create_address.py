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
def ex_create_address(self, name, region=None, address=None, description=None, address_type='EXTERNAL', subnetwork=None):
    """
        Create a static address in a region, or a global address.

        :param  name: Name of static address
        :type   name: ``str``

        :keyword  region: Name of region for the address (e.g. 'us-central1')
                          Use 'global' to create a global address.
        :type     region: ``str`` or :class:`GCERegion`

        :keyword  address: Ephemeral IP address to promote to a static one
                           (e.g. 'xxx.xxx.xxx.xxx')
        :type     address: ``str`` or ``None``

        :keyword  description: Optional descriptive comment.
        :type     description: ``str`` or ``None``

        :keyword  address_type: Optional The type of address to reserve,
                                either INTERNAL or EXTERNAL. If unspecified,
                                defaults to EXTERNAL.
        :type     description: ``str``

        :keyword  subnetwork: Optional The URL of the subnetwork in which to
                              reserve the address. If an IP address is
                              specified, it must be within the subnetwork's
                              IP range. This field can only be used with
                              INTERNAL type with GCE_ENDPOINT/DNS_RESOLVER
                              purposes.
        :type     description: ``str``

        :return:  Static Address object
        :rtype:   :class:`GCEAddress`
        """
    region = region or self.region
    if region is None:
        raise ValueError('REGION_NOT_SPECIFIED', 'Region must be provided for an address')
    if region != 'global' and (not hasattr(region, 'name')):
        region = self.ex_get_region(region)
    address_data = {'name': name}
    if address:
        address_data['address'] = address
    if description:
        address_data['description'] = description
    if address_type:
        if address_type not in ['EXTERNAL', 'INTERNAL']:
            raise ValueError('ADDRESS_TYPE_WRONG', 'Address type must be either EXTERNAL or                                  INTERNAL')
        else:
            address_data['addressType'] = address_type
    if subnetwork and address_type != 'INTERNAL':
        raise ValueError('INVALID_ARGUMENT_COMBINATION', 'Address type must be internal if subnetwork                              provided')
    if subnetwork and (not hasattr(subnetwork, 'name')):
        subnetwork = self.ex_get_subnetwork(subnetwork, region)
    if region == 'global':
        request = '/global/addresses'
    else:
        request = '/regions/%s/addresses' % region.name
    self.connection.async_request(request, method='POST', data=address_data)
    return self.ex_get_address(name, region=region)