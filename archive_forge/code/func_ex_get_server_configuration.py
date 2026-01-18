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
def ex_get_server_configuration(self):
    """

        Description: Server configuration and environment information
        Authentication: guest, untrusted or trusted
        Operation: sync
        Return: Dict representing server state

        The returned configuration depends on whether the connection
        is trusted or not
        :rtype: :class: .LXDServerInfo

        """
    response = self.connection.request('/%s' % self.version)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=200)
    meta = response_dict['metadata']
    return LXDServerInfo.build_from_response(metadata=meta)