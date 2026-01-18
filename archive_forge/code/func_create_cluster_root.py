from troveclient import base
from troveclient import common
from troveclient.v1 import users
def create_cluster_root(self, cluster, root_password=None):
    """Implements root-enable for clusters."""
    return self._enable_root(self.clusters_url % base.getid(cluster), root_password)