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
def ex_list_route_tables(self, route_table_ids=None, filters=None):
    """
        Describes one or more of a VPC's route tables.
        These are used to determine where network traffic is directed.

        :param      route_table_ids: Returns only route tables matching the
                                provided route table IDs. If not specified,
                                a list of all the route tables in the
                                corresponding region is returned.
        :type       route_table_ids: ``list``

        :param      filters: The filters so that the list returned includes
                             information for certain route tables only.
        :type       filters: ``dict``

        :rtype: ``list`` of :class:`.EC2RouteTable`
        """
    params = {'Action': 'DescribeRouteTables'}
    if route_table_ids:
        params.update(self._pathlist('RouteTableId', route_table_ids))
    if filters:
        params.update(self._build_filters(filters))
    response = self.connection.request(self.path, params=params)
    return self._to_route_tables(response.object)