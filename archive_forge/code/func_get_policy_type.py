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
def get_policy_type(self, policy_type):
    """Get the details about a policy type.

        :param policy_type: The name of a poicy_type or an object of
            :class:`~openstack.clustering.v1.policy_type.PolicyType`.

        :returns: A :class:`~openstack.clustering.v1.policy_type.PolicyType`
            object.
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            policy_type matching the name could be found.
        """
    return self._get(_policy_type.PolicyType, policy_type)