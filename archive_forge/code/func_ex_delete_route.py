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
def ex_delete_route(self, route_table, cidr):
    """
        Deletes a route entry from the route table.

        :param      route_table: The route table to delete the route from.
        :type       route_table: :class:`.EC2RouteTable`

        :param      cidr: The CIDR block used for the destination match.
        :type       cidr: ``str``

        :rtype:     ``bool``
        """
    params = {'Action': 'DeleteRoute', 'RouteTableId': route_table.id, 'DestinationCidrBlock': cidr}
    res = self.connection.request(self.path, params=params).object
    return self._get_boolean(res)