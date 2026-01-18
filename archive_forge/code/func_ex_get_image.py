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
def ex_get_image(self, fingerprint):
    """
        Returns a container image from the given image fingerprint

        :param fingerprint: image fingerprint
        :type  fingerprint: ``str``

        :rtype: :class:`.ContainerImage`
        """
    req = '/{}/images/{}'.format(self.version, fingerprint)
    response = self.connection.request(req)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=200)
    return self._to_image(metadata=response_dict['metadata'])