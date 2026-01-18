import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
def ex_get_storage_pool_volume(self, pool_id, type, name):
    """
        Description: information about a storage volume
        of a given type on a storage pool
        Introduced: with API extension storage
        Authentication: trusted
        Operation: sync
        Return: A StorageVolume  representing a storage volume
        """
    req = '/{}/storage-pools/{}/volumes/{}/{}'.format(self.version, pool_id, type, name)
    response = self.connection.request(req)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=200)
    return self._to_storage_volume(pool_id=pool_id, metadata=response_dict['metadata'])