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
def ex_authorize_security_group_permissive(self, name):
    """
        Edit a Security Group to allow all traffic.

        @note: This is a non-standard extension API, and only works for EC2.

        :param      name: The name of the security group to edit
        :type       name: ``str``

        :rtype: ``list`` of ``str``
        """
    results = []
    params = {'Action': 'AuthorizeSecurityGroupIngress', 'GroupName': name, 'IpProtocol': 'tcp', 'FromPort': '0', 'ToPort': '65535', 'CidrIp': '0.0.0.0/0'}
    try:
        results.append(self.connection.request(self.path, params=params.copy()).object)
    except Exception as e:
        if e.args[0].find('InvalidPermission.Duplicate') == -1:
            raise e
    params['IpProtocol'] = 'udp'
    try:
        results.append(self.connection.request(self.path, params=params.copy()).object)
    except Exception as e:
        if e.args[0].find('InvalidPermission.Duplicate') == -1:
            raise e
    params.update({'IpProtocol': 'icmp', 'FromPort': '-1', 'ToPort': '-1'})
    try:
        results.append(self.connection.request(self.path, params=params.copy()).object)
    except Exception as e:
        if e.args[0].find('InvalidPermission.Duplicate') == -1:
            raise e
    return results