import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def ex_destroy_service(self, service_id):
    """
        Destroy a service by ID

        http://docs.rancher.com/rancher/v1.2/en/api/api-resources/service/#delete

        :param service_id: The service to be destroyed.
        :type service_id: ``str``

        :return: True if destroy was successful, False otherwise.
        :rtype: ``bool``
        """
    result = self.connection.request('{}/services/{}'.format(self.baseuri, service_id), method='DELETE')
    return result.status in VALID_RESPONSE_CODES