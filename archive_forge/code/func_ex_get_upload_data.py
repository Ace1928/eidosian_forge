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
def ex_get_upload_data(self, container_id):
    """
        Retrieve information used for uploading files (upload url, auth token,
        etc).

        :rype: ``dict``
        """
    params = {}
    params['bucketId'] = container_id
    response = self.connection.request(action='b2_get_upload_url', method='GET', params=params)
    return response.object