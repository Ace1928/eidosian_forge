import os
import re
import shlex
import base64
import datetime
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def ex_delete_image(self, image):
    """
        Remove image from the filesystem

        :param  image: The image to remove
        :type   image: :class:`libcloud.container.base.ContainerImage`

        :rtype: ``bool``
        """
    result = self.connection.request('/v{}/images/{}'.format(self.version, image.name), method='DELETE')
    return result.status in VALID_RESPONSE_CODES