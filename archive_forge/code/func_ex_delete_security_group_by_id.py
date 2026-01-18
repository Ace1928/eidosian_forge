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
def ex_delete_security_group_by_id(self, group_id):
    """
        Deletes a new Security Group using the group ID.

        :param      group_id: The ID of the security group
        :type       group_id: ``str``

        :rtype: ``bool``
        """
    params = {'Action': 'DeleteSecurityGroup', 'GroupId': group_id}
    res = self.connection.request(self.path, params=params).object
    return self._get_boolean(res)