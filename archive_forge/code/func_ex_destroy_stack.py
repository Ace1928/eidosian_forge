import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def ex_destroy_stack(self, env_id):
    """
        Destroy a stack by ID

        http://docs.rancher.com/rancher/v1.2/en/api/api-resources/environment/#delete

        :param env_id: The stack to be destroyed.
        :type env_id: ``str``

        :return: True if destroy was successful, False otherwise.
        :rtype: ``bool``
        """
    result = self.connection.request('{}/environments/{}'.format(self.baseuri, env_id), method='DELETE')
    return result.status in VALID_RESPONSE_CODES