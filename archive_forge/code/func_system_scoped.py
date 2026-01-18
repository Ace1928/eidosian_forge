import functools
from keystoneauth1 import _utils as utils
from keystoneauth1.access import service_catalog
from keystoneauth1.access import service_providers
@_missingproperty
def system_scoped(self):
    return self._data['token']['system'].get('all', False)