import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def start_container(self, container):
    """
        Start a container

        :param container: The container to be started
        :type  container: :class:`libcloud.container.base.Container`

        :return: The container refreshed with current data
        :rtype: :class:`libcloud.container.base.Container`
        """
    result = self.connection.request('{}/containers/{}?action=start'.format(self.baseuri, container.id), method='POST').object
    return self._to_container(result)