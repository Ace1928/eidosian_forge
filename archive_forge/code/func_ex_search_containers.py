import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def ex_search_containers(self, search_params):
    """
        Search for containers matching certain filters

        i.e. ``{ "imageUuid": "docker:mysql", "state": "running"}``

        :param search_params: A collection of search parameters to use.
        :type search_params: ``dict``

        :rtype: ``list``
        """
    search_list = []
    for f, v in search_params.items():
        search_list.append(f + '=' + v)
    search_items = '&'.join(search_list)
    result = self.connection.request('{}/containers?{}'.format(self.baseuri, search_items)).object
    return result['data']