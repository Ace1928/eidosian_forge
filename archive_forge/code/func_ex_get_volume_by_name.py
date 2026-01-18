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
def ex_get_volume_by_name(self, name, vol_type='custom'):
    """
        Returns a storage volume that has the given name.
        The function will loop over all storage-polls available
        and will pick the first volume from the first storage poll
        that matches the given name. Thus this function can be
        quite expensive

        :param name: The name of the volume to look for
        :type  name: str

        :param vol_type: The type of the volume default is custom
        :type  vol_type: str

        :return: A StorageVolume  representing a storage volume
        """
    req = '/%s/storage-pools' % self.version
    response = self.connection.request(req)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=200)
    pools = response_dict['metadata']
    for pool in pools:
        pool_id = pool.split('/')[-1]
        volumes = self.ex_list_storage_pool_volumes(pool_id=pool_id)
        for vol in volumes:
            if vol.name == name:
                return vol
    return None