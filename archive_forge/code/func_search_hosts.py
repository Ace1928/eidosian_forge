import functools
from openstack.cloud import _utils
from openstack.config import loader
from openstack import connection
from openstack import exceptions
def search_hosts(self, name_or_id=None, filters=None, expand=True):
    hosts = self.list_hosts(expand=expand)
    return _utils._filter_list(hosts, name_or_id, filters)