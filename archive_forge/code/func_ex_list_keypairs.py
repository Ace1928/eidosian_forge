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
def ex_list_keypairs(self):
    """
        Lists all the keypair names and fingerprints.

        :rtype: ``list`` of ``dict``
        """
    warnings.warn('This method has been deprecated in favor of list_key_pairs method')
    key_pairs = self.list_key_pairs()
    result = []
    for key_pair in key_pairs:
        item = {'keyName': key_pair.name, 'keyFingerprint': key_pair.fingerprint}
        result.append(item)
    return result