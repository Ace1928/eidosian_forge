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
def get_profile_type(self, profile_type):
    """Get the details about a profile type.

        :param profile_type: The name of the profile_type to retrieve or an
            object of
            :class:`~openstack.clustering.v1.profile_type.ProfileType`.

        :returns: A :class:`~openstack.clustering.v1.profile_type.ProfileType`
            object.
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            profile_type matching the name could be found.
        """
    return self._get(_profile_type.ProfileType, profile_type)