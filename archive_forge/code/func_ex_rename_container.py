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
def ex_rename_container(self, container, name):
    """
        Rename a container

        :param container: The container to be renamed
        :type  container: :class:`libcloud.container.base.Container`

        :param name: The new name
        :type  name: ``str``

        :rtype: :class:`libcloud.container.base.Container`
        """
    result = self.connection.request('/v{}/containers/{}/rename?name={}'.format(self.version, container.id, name), method='POST')
    if result.status in VALID_RESPONSE_CODES:
        return self.get_container(container.id)