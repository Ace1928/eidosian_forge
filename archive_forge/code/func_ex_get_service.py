import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def ex_get_service(self, service_id):
    """
        Get a service by ID

        :param service_id: The service_id to be obtained.
        :type service_id: ``str``

        :rtype: ``dict``
        """
    result = self.connection.request('{}/services/{}'.format(self.baseuri, service_id)).object
    return result