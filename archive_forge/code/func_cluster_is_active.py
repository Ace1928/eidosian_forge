from openstack import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients.os import openstacksdk as sdk_plugin
from heat.engine import constraints
def cluster_is_active(self, cluster_id):
    cluster = self.client().get_cluster(cluster_id)
    if cluster.status == 'ACTIVE':
        return True
    elif cluster.status == 'ERROR':
        raise exception.ResourceInError(status_reason=cluster.status_reason, resource_status=cluster.status)
    return False