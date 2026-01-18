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
def ex_describe_addresses_for_node(self, node):
    """
        Returns a list of Elastic IP Addresses associated with this node.

        :param      node: Node instance
        :type       node: :class:`Node`

        :return: List Elastic IP Addresses attached to this node.
        :rtype: ``list`` of ``str``
        """
    node_elastic_ips = self.ex_describe_addresses([node])
    return node_elastic_ips[node.id]