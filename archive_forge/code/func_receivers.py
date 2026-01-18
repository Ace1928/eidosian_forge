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
def receivers(self, **query):
    """Retrieve a generator of receivers.

        :param kwargs query: Optional query parameters for restricting the
            receivers to be returned. Available parameters include:

            * name: The name of a receiver object.
            * type: The type of receiver objects.
            * cluster_id: The ID of the associated cluster.
            * action: The name of the associated action.
            * sort: A list of sorting keys separated by commas. Each sorting
              key can optionally be attached with a sorting direction
              modifier which can be ``asc`` or ``desc``.
            * global_project: A boolean value indicating whether receivers
            *   from all projects will be returned.

        :returns: A generator of receiver instances.
        """
    return self._list(_receiver.Receiver, **query)