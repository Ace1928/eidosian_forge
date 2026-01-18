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
def ex_detach_drive(self, node, drive):
    data = self.ex_get_node(node.id, return_json=True)
    data['drives'] = [item for item in data['drives'] if item['drive']['uuid'] != drive.id]
    action = '/servers/%s/' % node.id
    response = self.connection.request(action=action, data=data, method='PUT')
    return response.status == 200