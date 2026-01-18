from libcloud.utils.py3 import ET, httplib
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.types import LibcloudError, MalformedResponseError, KeyPairDoesNotExistError
from libcloud.common.exceptions import BaseHTTPError
from libcloud.common.openstack_identity import (
def has_content_type(self, content_type):
    content_type_value = self.headers.get('content-type') or ''
    content_type_value = content_type_value.lower()
    return content_type_value.find(content_type.lower()) > -1