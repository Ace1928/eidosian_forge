from collections import defaultdict
from collections import namedtuple
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from openstack import connection
from oslo_config import cfg
from oslo_log import log
from oslo_limit import exception
from oslo_limit import opts
@staticmethod
def enforce_limits(project_id, limits, current_usage, deltas):
    """Check that proposed usage is not over given limits

        :param project_id: project being checked or None
        :param limits: list of (resource_name,limit) pairs
        :param current_usage: dict of resource name and current usage
        :param deltas: dict of resource name and proposed additional usage

        :raises exception.ClaimExceedsLimit: raise if over limit
        """
    over_limit_list = []
    for resource_name, limit in limits:
        if resource_name not in current_usage:
            msg = 'unable to get current usage for %s' % resource_name
            raise ValueError(msg)
        current = int(current_usage[resource_name])
        delta = int(deltas[resource_name])
        proposed_usage_total = current + delta
        if proposed_usage_total > limit:
            over_limit_list.append(exception.OverLimitInfo(resource_name, limit, current, delta))
    if len(over_limit_list) > 0:
        LOG.debug('hit limit for project: %s', over_limit_list)
        raise exception.ProjectOverLimit(project_id, over_limit_list)