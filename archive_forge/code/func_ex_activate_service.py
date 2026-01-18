import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def ex_activate_service(self, service_id):
    """
        Activate a service.

        http://docs.rancher.com/rancher/v1.2/en/api/api-resources/service/#activate

        :param service_id: The service to activate services for.
        :type service_id: ``str``

        :return: True if activate was successful, False otherwise.
        :rtype: ``bool``
        """
    result = self.connection.request('{}/services/{}?action=activate'.format(self.baseuri, service_id), method='POST')
    return result.status in VALID_RESPONSE_CODES