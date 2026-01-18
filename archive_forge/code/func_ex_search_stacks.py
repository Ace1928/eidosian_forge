import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
def ex_search_stacks(self, search_params):
    """
        Search for stacks matching certain filters

        i.e. ``{ "name": "awesomestack"}``

        :param search_params: A collection of search parameters to use.
        :type search_params: ``dict``

        :rtype: ``list``
        """
    search_list = []
    for f, v in search_params.items():
        search_list.append(f + '=' + v)
    search_items = '&'.join(search_list)
    result = self.connection.request('{}/environments?{}'.format(self.baseuri, search_items)).object
    return result['data']