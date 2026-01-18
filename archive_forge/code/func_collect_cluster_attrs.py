from openstack.clustering.v1 import action as _action
from openstack.clustering.v1 import build_info
from openstack.clustering.v1 import cluster as _cluster
from openstack.clustering.v1 import cluster_attr as _cluster_attr
from openstack.clustering.v1 import cluster_policy as _cluster_policy
from openstack.clustering.v1 import event as _event
from openstack.clustering.v1 import node as _node
from openstack.clustering.v1 import policy as _policy
from openstack.clustering.v1 import policy_type as _policy_type
from openstack.clustering.v1 import profile as _profile
from openstack.clustering.v1 import profile_type as _profile_type
from openstack.clustering.v1 import receiver as _receiver
from openstack.clustering.v1 import service as _service
from openstack import proxy
from openstack import resource
def collect_cluster_attrs(self, cluster, path, **query):
    """Collect attribute values across a cluster.

        :param cluster: The value can be either the ID of a cluster or a
            :class:`~openstack.clustering.v1.cluster.Cluster` instance.
        :param path: A Json path string specifying the attribute to collect.
        :param query: Optional query parameters to be sent to limit the
            resources being returned.

        :returns: A dictionary containing the list of attribute values.
        """
    return self._list(_cluster_attr.ClusterAttr, cluster_id=cluster, path=path)