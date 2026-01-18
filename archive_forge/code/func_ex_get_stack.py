import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def ex_get_stack(self, env_id):
    """
        Get a stack by ID

        :param env_id: The stack to be obtained.
        :type env_id: ``str``

        :rtype: ``dict``
        """
    result = self.connection.request('{}/environments/{}'.format(self.baseuri, env_id)).object
    return result