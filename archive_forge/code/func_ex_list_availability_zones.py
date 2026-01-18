import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
def ex_list_availability_zones(self, only_available=True):
    """
        Returns a list of :class:`ExEC2AvailabilityZone` objects for the
        current region.

        Note: This is an extension method and is only available for EC2
        driver.

        :keyword  only_available: If true, returns only availability zones
                                  with state 'available'
        :type     only_available: ``str``

        :rtype: ``list`` of :class:`ExEC2AvailabilityZone`
        """
    params = {'Action': 'DescribeAvailabilityZones'}
    filters = {'region-name': self.region_name}
    if only_available:
        filters['state'] = 'available'
    params.update(self._build_filters(filters))
    result = self.connection.request(self.path, params=params.copy()).object
    availability_zones = []
    for element in findall(element=result, xpath='availabilityZoneInfo/item', namespace=NAMESPACE):
        name = findtext(element=element, xpath='zoneName', namespace=NAMESPACE)
        zone_state = findtext(element=element, xpath='zoneState', namespace=NAMESPACE)
        region_name = findtext(element=element, xpath='regionName', namespace=NAMESPACE)
        availability_zone = ExEC2AvailabilityZone(name=name, zone_state=zone_state, region_name=region_name)
        availability_zones.append(availability_zone)
    return availability_zones