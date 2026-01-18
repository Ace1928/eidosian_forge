import functools
from keystoneauth1 import _utils as utils
from keystoneauth1.access import service_catalog
from keystoneauth1.access import service_providers
@property
def domain_scoped(self):
    try:
        return bool(self._domain)
    except KeyError:
        return False