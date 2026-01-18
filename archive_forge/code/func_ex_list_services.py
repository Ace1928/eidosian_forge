import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def ex_list_services(self):
    """
        List all Rancher Services

        http://docs.rancher.com/rancher/v1.2/en/api/api-resources/service/

        :rtype: ``list`` of ``dict``
        """
    result = self.connection.request('%s/services' % self.baseuri).object
    return result['data']