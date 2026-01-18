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
def ex_modify_instance_keypair(self, instance_id, key_name=None):
    """
        Modifies the keypair associated with a specified instance.
        Once the modification is done, you must restart the instance.

        :param      instance_id: The ID of the instance
        :type       instance_id: ``string``

        :param      key_name: The name of the keypair
        :type       key_name: ``string``
        """
    params = {'Action': 'ModifyInstanceKeypair'}
    params.update({'instanceId': instance_id})
    if key_name is not None:
        params.update({'keyName': key_name})
    response = self.connection.request(self.path, params=params, method='GET').object
    return findtext(element=response, xpath='return', namespace=OUTSCALE_NAMESPACE) == 'true'