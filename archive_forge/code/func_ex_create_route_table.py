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
def ex_create_route_table(self, network, name=None):
    """
        Creates a route table within a VPC.

        :param      vpc_id: The VPC that the subnet should be created in.
        :type       vpc_id: :class:`.EC2Network`

        :rtype:     :class: `.EC2RouteTable`
        """
    params = {'Action': 'CreateRouteTable', 'VpcId': network.id}
    response = self.connection.request(self.path, params=params).object
    element = response.findall(fixxpath(xpath='routeTable', namespace=NAMESPACE))[0]
    route_table = self._to_route_table(element, name=name)
    if name and self.ex_create_tags(route_table, {'Name': name}):
        route_table.extra['tags']['Name'] = name
    return route_table