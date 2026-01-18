from collections import defaultdict
from collections import namedtuple
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from openstack import connection
from oslo_config import cfg
from oslo_log import log
from oslo_limit import exception
from oslo_limit import opts
def get_project_limits(self, project_id, resource_names):
    """Get all the limits for given project a resource_name list

        If a limit is not found, it will be considered to be zero
        (i.e. no quota)

        :param project_id: project being checked or None
        :param resource_names: list of resource_name strings
        :return: list of (resource_name,limit) pairs
        """
    project_limits = []
    for resource_name in resource_names:
        try:
            limit = self._get_limit(project_id, resource_name)
        except _LimitNotFound:
            limit = 0
        project_limits.append((resource_name, limit))
    return project_limits