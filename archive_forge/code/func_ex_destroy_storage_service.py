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
def ex_destroy_storage_service(self, name):
    """
        Destroy storage service. Storage service must not have any active
        blobs. Sometimes Azure likes to hold onto volumes after they are
        deleted for an inordinate amount of time, so sleep before calling
        this method after volume deletion.

        :param name: Name of storage service.
        :type  name: ``str``

        :rtype: ``bool``
        """
    response = self._perform_storage_service_delete(self._get_storage_service_path(name))
    self.raise_for_response(response, 200)
    return True