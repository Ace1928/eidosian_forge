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
def ex_modify_volume(self, volume, parameters):
    """
        Modify volume parameters.
        A list of valid parameters can be found at https://goo.gl/N0rPEQ

        :param      volume: Volume instance
        :type       volume: :class:`Volume`

        :param      parameters: Dictionary with updated volume parameters
        :type       parameters: ``dict``

        :return: Volume modification status object
        :rtype: :class:`VolumeModification
        """
    parameters = parameters or {}
    volume_type = parameters.get('VolumeType')
    if volume_type and volume_type not in VALID_VOLUME_TYPES:
        raise ValueError('Invalid volume type specified: %s' % volume_type)
    parameters.update({'Action': 'ModifyVolume', 'VolumeId': volume.id})
    response = self.connection.request(self.path, params=parameters.copy()).object
    return self._to_volume_modification(response.findall(fixxpath(xpath='volumeModification', namespace=NAMESPACE))[0])