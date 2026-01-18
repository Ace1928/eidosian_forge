from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
def search_security_groups(self, name_or_id=None, filters=None):
    groups = self.list_security_groups(filters=filters if isinstance(filters, dict) else None)
    return _utils._filter_list(groups, name_or_id, filters)