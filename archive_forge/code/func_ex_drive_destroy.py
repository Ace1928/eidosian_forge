import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
def ex_drive_destroy(self, drive_uuid):
    """
        Destroy a drive with a specified uuid.
        If the drive is currently mounted an exception is thrown.

        :param      drive_uuid: Drive uuid which should be used
        :type       drive_uuid: ``str``

        :rtype: ``bool``
        """
    response = self.connection.request(action='/drives/%s/destroy' % drive_uuid, method='POST')
    return response.status == 204