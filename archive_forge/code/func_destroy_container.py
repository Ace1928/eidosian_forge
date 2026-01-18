import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def destroy_container(self, container):
    """
        Remove a container

        :param container: The container to be destroyed
        :type  container: :class:`libcloud.container.base.Container`

        :return: True if the destroy was successful, False otherwise.
        :rtype: ``bool``
        """
    result = self.connection.request('{}/containers/{}'.format(self.baseuri, container.id), method='DELETE').object
    return self._to_container(result)