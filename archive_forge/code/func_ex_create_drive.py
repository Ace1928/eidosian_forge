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
def ex_create_drive(self, name, size, media='disk', ex_avoid=None):
    """
        Create a new drive.

        :param name: Drive name.
        :type name: ``str``

        :param size: Drive size in GBs.
        :type size: ``int``

        :param media: Drive media type (cdrom, disk).
        :type media: ``str``

        :param ex_avoid: A list of other drive uuids to avoid when
                         creating this drive. If provided, drive will
                         attempt to be created on a different
                         physical infrastructure from other drives
                         specified using this argument. (optional)
        :type ex_avoid: ``list``

        :return: Created drive object.
        :rtype: :class:`.CloudSigmaDrive`
        """
    params = {}
    data = {'name': name, 'size': size * 1024 * 1024 * 1024, 'media': media}
    if ex_avoid:
        params['avoid'] = ','.join(ex_avoid)
    action = '/drives/'
    response = self.connection.request(action=action, method='POST', params=params, data=data).object
    drive = self._to_drive(data=response['objects'][0])
    return drive