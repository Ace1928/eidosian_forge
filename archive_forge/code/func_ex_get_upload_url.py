import base64
import hashlib
from libcloud.utils.py3 import b, next, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks, exhaust_iterator
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.utils.escape import sanitize_object_name
from libcloud.storage.types import ObjectDoesNotExistError, ContainerDoesNotExistError
from libcloud.storage.providers import Provider
def ex_get_upload_url(self, container_id):
    """
        Retrieve URL used for file uploads.

        :rtype: ``str``
        """
    result = self.ex_get_upload_data(container_id=container_id)
    upload_url = result['uploadUrl']
    return upload_url