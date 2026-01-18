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
def policies(self, **query):
    """Retrieve a generator of policies.

        :param kwargs query: Optional query parameters to be sent to
            restrict the policies to be returned. Available parameters include:

            * name: The name of a policy.
            * type: The type name of a policy.
            * sort: A list of sorting keys separated by commas. Each sorting
              key can optionally be attached with a sorting direction
              modifier which can be ``asc`` or ``desc``.
            * limit: Requests a specified size of returned items from the
              query.  Returns a number of items up to the specified limit
              value.
            * marker: Specifies the ID of the last-seen item. Use the limit
              parameter to make an initial limited request and use the ID of
              the last-seen item from the response as the marker parameter
              value in a subsequent limited request.
            * global_project: A boolean value indicating whether policies from
              all projects will be returned.

        :returns: A generator of policy instances.
        """
    return self._list(_policy.Policy, **query)