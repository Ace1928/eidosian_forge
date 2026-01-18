import functools
from keystoneauth1 import _utils as utils
from keystoneauth1.access import service_catalog
from keystoneauth1.access import service_providers
def has_service_catalog(self):
    return 'catalog' in self._data['token']