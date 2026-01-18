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
def get_resource_path_api_2(self):
    """
        This method returns a resource path which is necessary for referencing
        resources that require a full path instead of just an ID, such as
        networks, and customer snapshots.
        """
    return '{}/{}/{}'.format(self.api_path_version_2, self.active_api_version, self._get_orgId())