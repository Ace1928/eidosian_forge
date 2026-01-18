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
def ex_get_drive(self, drive_id):
    """
        Retrieve information about a single drive.

        :param drive_id: ID of the drive to retrieve.
        :type drive_id: ``str``

        :return: Drive object.
        :rtype: :class:`.CloudSigmaDrive`
        """
    action = '/drives/%s/' % drive_id
    response = self.connection.request(action=action).object
    drive = self._to_drive(data=response)
    return drive