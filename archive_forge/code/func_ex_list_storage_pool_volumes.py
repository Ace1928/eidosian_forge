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
def ex_list_storage_pool_volumes(self, pool_id, detailed=True):
    """
        Description: list of storage volumes
        associated with the given storage pool

        :param pool_id: the id of the storage pool to query
        :param detailed: boolean flag.
        If True extra API calls are made to fill in the missing details
                                       of the storage volumes

        Authentication: trusted
        Operation: sync
        Return: list of storage volumes that
        currently exist on a given storage pool

        :rtype: A list of :class: StorageVolume
        """
    req = '/{}/storage-pools/{}/volumes'.format(self.version, pool_id)
    response = self.connection.request(req)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=200)
    volumes = []
    for volume in response_dict['metadata']:
        volume = volume.split('/')
        name = volume[-1]
        type = volume[-2]
        if not detailed:
            metadata = {'config': {'size': None}, 'name': name, 'type': type, 'used_by': None}
            volumes.append(self._to_storage_volume(pool_id=pool_id, metadata=metadata))
        else:
            volume = self.ex_get_storage_pool_volume(pool_id=pool_id, type=type, name=name)
            volumes.append(volume)
    return volumes