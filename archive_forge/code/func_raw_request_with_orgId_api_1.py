import re
import xml.etree.ElementTree as etree
from io import BytesIO
from copy import deepcopy
from time import sleep
from base64 import b64encode
from typing import Dict
from functools import wraps
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
def raw_request_with_orgId_api_1(self, action, params=None, data='', headers=None, method='GET'):
    action = '{}/{}'.format(self.get_resource_path_api_1(), action)
    return super().request(action=action, params=params, data=data, method=method, headers=headers, raw=True)