from collections import defaultdict
from collections import namedtuple
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from openstack import connection
from oslo_config import cfg
from oslo_log import log
from oslo_limit import exception
from oslo_limit import opts
def get_registered_limits(self, resource_names):
    """Get all the default limits for a given resource name list

        :param resource_names: list of resource_name strings
        :return: list of (resource_name, limit) pairs
        """
    registered_limits = []
    for resource_name in resource_names:
        reg_limit = self._get_registered_limit(resource_name)
        if reg_limit:
            limit = reg_limit.default_limit
        else:
            limit = 0
        registered_limits.append((resource_name, limit))
    return registered_limits