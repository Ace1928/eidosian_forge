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
def ex_revoke_security_group_egress(self, id, from_port, to_port, cidr_ips=None, group_pairs=None, protocol='tcp'):
    """
        Edit a Security Group to revoke specific egress traffic using
        CIDR blocks or either a group ID, group name or user ID (account).
        This call is not supported for EC2 classic and only works for
        VPC groups.

        :param      id: The id of the security group to edit
        :type       id: ``str``

        :param      from_port: The beginning of the port range to open
        :type       from_port: ``int``

        :param      to_port: The end of the port range to open
        :type       to_port: ``int``

        :param      cidr_ips: The list of ip ranges to allow traffic for.
        :type       cidr_ips: ``list``

        :param      group_pairs: Source user/group pairs to allow traffic for.
                    More info can be found at http://goo.gl/stBHJF

                    EC2 Classic Example: To allow access from any system
                    associated with the default group on account 1234567890

                    [{'group_name': 'default', 'user_id': '1234567890'}]

                    VPC Example: Allow access from any system associated with
                    security group sg-47ad482e on your own account

                    [{'group_id': ' sg-47ad482e'}]
        :type       group_pairs: ``list`` of ``dict``

        :param      protocol: tcp/udp/icmp
        :type       protocol: ``str``

        :rtype: ``bool``
        """
    params = self._get_common_security_group_params(id, protocol, from_port, to_port, cidr_ips, group_pairs)
    params['Action'] = 'RevokeSecurityGroupEgress'
    res = self.connection.request(self.path, params=params).object
    return self._get_boolean(res)