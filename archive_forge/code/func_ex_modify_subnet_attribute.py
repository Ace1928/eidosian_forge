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
def ex_modify_subnet_attribute(self, subnet, attribute='auto_public_ip', value=False):
    """
        Modifies a subnet attribute.
        You can only modify one attribute at a time.

        :param      subnet: The subnet to delete
        :type       subnet: :class:`.EC2NetworkSubnet`

        :param      attribute: The attribute to set on the subnet; one of:
                               ``'auto_public_ip'``: Automatically allocate a
                               public IP address when a server is created
                               ``'auto_ipv6'``: Automatically assign an IPv6
                               address when a server is created
        :type       attribute: ``str``

        :param      value: The value to set the subnet attribute to
                           (defaults to ``False``)
        :type       value: ``bool``

        :rtype:     ``bool``
        """
    params = {'Action': 'ModifySubnetAttribute', 'SubnetId': subnet.id}
    if attribute == 'auto_public_ip':
        params['MapPublicIpOnLaunch.Value'] = value
    elif attribute == 'auto_ipv6':
        params['AssignIpv6AddressOnCreation.Value'] = value
    else:
        raise ValueError('Unsupported attribute: %s' % attribute)
    res = self.connection.request(self.path, params=params).object
    return self._get_boolean(res)