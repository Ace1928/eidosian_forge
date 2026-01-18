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
def ex_create_subscription(self, amount, period, resource, auto_renew=False):
    """
        Create a new subscription.

        :param amount: Subscription amount. For example, in dssd case this
                       would be disk size in gigabytes.
        :type amount: ``int``

        :param period: Subscription period. For example: 30 days, 1 week, 1
                                            month, ...
        :type period: ``str``

        :param resource: Resource the purchase the subscription for.
        :type resource: ``str``

        :param auto_renew: True to automatically renew the subscription.
        :type auto_renew: ``bool``
        """
    data = [{'amount': amount, 'period': period, 'auto_renew': auto_renew, 'resource': resource}]
    response = self.connection.request(action='/subscriptions/', data=data, method='POST')
    data = response.object['objects'][0]
    subscription = self._to_subscription(data=data)
    return subscription