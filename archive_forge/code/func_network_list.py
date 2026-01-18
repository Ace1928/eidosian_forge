from urllib import parse
from zunclient import api_versions
from zunclient.common import base
from zunclient.common import utils
from zunclient import exceptions
def network_list(self, container):
    return self._list(self._path(container) + '/network_list', 'networks')