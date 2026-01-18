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
def ex_create_subnet(self, vpc_id, cidr_block, availability_zone, name=None):
    """
        Creates a network subnet within a VPC.

        :param      vpc_id: The ID of the VPC that the subnet should be
                            associated with
        :type       vpc_id: ``str``

        :param      cidr_block: The CIDR block assigned to the subnet
        :type       cidr_block: ``str``

        :param      availability_zone: The availability zone where the subnet
                                       should reside
        :type       availability_zone: ``str``

        :param      name: An optional name for the network
        :type       name: ``str``

        :rtype:     :class: `EC2NetworkSubnet`
        """
    params = {'Action': 'CreateSubnet', 'VpcId': vpc_id, 'CidrBlock': cidr_block, 'AvailabilityZone': availability_zone}
    response = self.connection.request(self.path, params=params).object
    element = response.findall(fixxpath(xpath='subnet', namespace=NAMESPACE))[0]
    subnet = self._to_subnet(element, name)
    if name and self.ex_create_tags(subnet, {'Name': name}):
        subnet.extra['tags']['Name'] = name
    return subnet