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
def ex_delete_firewall_policy(self, policy):
    """
        Delete a firewall policy.

        :param policy: Policy to delete to.
        :type policy: :class:`.CloudSigmaFirewallPolicy`

        :return: ``True`` on success, ``False`` otherwise.
        :rtype: ``bool``
        """
    action = '/fwpolicies/%s/' % policy.id
    response = self.connection.request(action=action, method='DELETE')
    return response.status == httplib.NO_CONTENT