import os
import hmac
import atexit
from time import time
from hashlib import sha1
from libcloud.utils.py3 import b, httplib, urlquote, urlencode
from libcloud.common.base import Response, RawResponse
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
from libcloud.common.openstack import OpenStackDriverMixin, OpenStackBaseConnection
from libcloud.common.rackspace import AUTH_URL
from libcloud.storage.providers import Provider
from io import FileIO as file
def ex_set_error_page(self, container, file_name='error.html'):
    """
        Set a custom error page which is displayed if file is not found and
        serving of a static website is enabled.

        :param container: Container instance
        :type container: :class:`Container`

        :param file_name: Name of the object which becomes the error page.
        :type file_name: ``str``

        :rtype: ``bool``
        """
    container_name = container.name
    headers = {'X-Container-Meta-Web-Error': file_name}
    response = self.connection.request('/%s' % container_name, method='POST', headers=headers, cdn_request=False)
    return response.status in [httplib.CREATED, httplib.ACCEPTED]