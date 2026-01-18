import ipaddress
import time
import warnings
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
from openstack import warnings as os_warnings
def search_floating_ips(self, id=None, filters=None):
    warnings.warn('search_floating_ips is deprecated. Use search_resource instead.', os_warnings.OpenStackDeprecationWarning)
    if self._use_neutron_floating() and isinstance(filters, dict):
        return list(self.network.ips(**filters))
    else:
        floating_ips = self.list_floating_ips()
        return _utils._filter_list(floating_ips, id, filters)