import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def ex_deactivate_stack(self, env_id):
    """
        Deactivate Services for a stack.

        http://docs.rancher.com/rancher/v1.2/en/api/api-resources/environment/#deactivateservices

        :param env_id: The stack to deactivate services for.
        :type env_id: ``str``

        :return: True if deactivate was successful, False otherwise.
        :rtype: ``bool``
        """
    result = self.connection.request('{}/environments/{}?action=deactivateservices'.format(self.baseuri, env_id), method='POST')
    return result.status in VALID_RESPONSE_CODES