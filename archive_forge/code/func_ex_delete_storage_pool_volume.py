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
def ex_delete_storage_pool_volume(self, pool_id, type, name):
    """
        Delete a storage volume of a given type on a given storage pool

        :param pool_id:
        :type ``str``

        :param type:
        :type  ``str``

        :param name:
        :type ``str``

        :return:
        """
    try:
        req = '/{}/storage-pools/{}/volumes/{}/{}'.format(self.version, pool_id, type, name)
        response = self.connection.request(req, method='DELETE')
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
    except BaseHTTPError as err:
        raise self._get_lxd_api_exception_for_error(err)
    return True