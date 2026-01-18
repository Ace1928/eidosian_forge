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
def ex_import_keypair(self, name, keyfile):
    """
        Imports a new public key where the public key is passed via a filename.

        @note: This is a non-standard extension API, and only works for EC2.

        :param      name: The name of the public key to import. This must be
                          unique, otherwise an InvalidKeyPair. Duplicate
                          exception is raised.
        :type       name: ``str``

        :param     keyfile: The filename with the path of the public key
                            to import.
        :type      keyfile: ``str``

        :rtype: ``dict``
        """
    warnings.warn('This method has been deprecated in favor of import_key_pair_from_file method')
    key_pair = self.import_key_pair_from_file(name=name, key_file_path=keyfile)
    result = {'keyName': key_pair.name, 'keyFingerprint': key_pair.fingerprint}
    return result