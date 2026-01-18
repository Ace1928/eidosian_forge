import re
import copy
import time
import base64
import random
import collections
from xml.dom import minidom
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from libcloud.utils.py3 import ET, httplib, urlparse
from libcloud.utils.py3 import urlquote as url_quote
from libcloud.utils.py3 import _real_unicode, ensure_string
from libcloud.utils.misc import ReprMixin
from libcloud.common.azure import AzureRedirectException, AzureServiceManagementConnection
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
def ex_add_instance_endpoints(self, node, endpoints, ex_deployment_slot='Production'):
    all_endpoints = [{'name': endpoint.name, 'protocol': endpoint.protocol, 'port': endpoint.public_port, 'local_port': endpoint.local_port} for endpoint in node.extra['instance_endpoints']]
    all_endpoints.extend(endpoints)
    result = self.ex_set_instance_endpoints(node, all_endpoints, ex_deployment_slot)
    return result