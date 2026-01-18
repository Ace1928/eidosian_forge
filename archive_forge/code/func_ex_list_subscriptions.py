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
def ex_list_subscriptions(self, status='all', resources=None):
    """
        List subscriptions for this account.

        :param status: Only return subscriptions with the provided status
                       (optional).
        :type status: ``str``
        :param resources: Only return subscriptions for the provided resources
                          (optional).
        :type resources: ``list``

        :rtype: ``list``
        """
    params = {}
    if status:
        params['status'] = status
    if resources:
        params['resource'] = ','.join(resources)
    response = self.connection.request(action='/subscriptions/', params=params).object
    subscriptions = self._to_subscriptions(data=response)
    return subscriptions