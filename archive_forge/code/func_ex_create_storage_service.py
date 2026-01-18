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
def ex_create_storage_service(self, name, location, description=None, affinity_group=None, extended_properties=None):
    """
        Create an azure storage service.

        :param      name: Name of the service to create
        :type       name: ``str``

        :param      location: Standard azure location string
        :type       location: ``str``

        :param      description: (Optional) Description of storage service.
        :type       description: ``str``

        :param      affinity_group: (Optional) Azure affinity group.
        :type       affinity_group: ``str``

        :param      extended_properties: (Optional) Additional configuration
                                         options support by Azure.
        :type       extended_properties: ``dict``

        :rtype: ``bool``
        """
    response = self._perform_storage_service_create(self._get_storage_service_path(), AzureXmlSerializer.create_storage_service_to_xml(service_name=name, label=self._encode_base64(name), description=description, location=location, affinity_group=affinity_group, extended_properties=extended_properties))
    self.raise_for_response(response, 202)
    return True