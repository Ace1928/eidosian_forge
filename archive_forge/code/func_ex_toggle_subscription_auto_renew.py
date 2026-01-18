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
def ex_toggle_subscription_auto_renew(self, subscription):
    """
        Toggle subscription auto renew status.

        :param subscription: Subscription to toggle the auto renew flag for.
        :type subscription: :class:`.CloudSigmaSubscription`

        :return: ``True`` on success, ``False`` otherwise.
        :rtype: ``bool``
        """
    path = '/subscriptions/%s/action/' % subscription.id
    response = self._perform_action(path=path, action='auto_renew', method='POST')
    return response.status == httplib.OK