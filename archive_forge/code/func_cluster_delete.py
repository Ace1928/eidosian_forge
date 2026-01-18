from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
def cluster_delete(self, cluster_id):
    try:
        cluster = self.client().clusters.get(cluster_id)
        cluster_status = cluster.task['name']
        if cluster_status not in self.DELETE_STATUSES:
            return False
        if cluster_status != self.DELETING:
            cluster.delete()
    except Exception as ex:
        self.client_plugin().ignore_not_found(ex)
    return True