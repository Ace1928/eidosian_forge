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
def ex_delete_keypair(self, keypair):
    """
        Deletes a key pair by name.

        @note: This is a non-standard extension API, and only works with EC2.

        :param      keypair: The name of the keypair to delete.
        :type       keypair: ``str``

        :rtype: ``bool``
        """
    warnings.warn('This method has been deprecated in favor of delete_key_pair method')
    keypair = KeyPair(name=keypair, public_key=None, fingerprint=None, driver=self)
    return self.delete_key_pair(keypair)