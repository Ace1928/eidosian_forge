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
def ex_list_processes(self, container):
    """
        List processes running inside a container

        :param container: The container to list processes for.
        :type  container: :class:`libcloud.container.base.Container`

        :rtype: ``str``
        """
    result = self.connection.request('/v{}/containers/{}/top'.format(self.version, container.id)).object
    return result