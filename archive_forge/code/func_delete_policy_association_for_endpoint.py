from keystoneclient import base
from keystoneclient.i18n import _
from keystoneclient.v3 import endpoints
from keystoneclient.v3 import policies
def delete_policy_association_for_endpoint(self, policy, endpoint):
    """Delete an association between a policy and an endpoint."""
    return self._act_on_policy_association_for_endpoint(policy, endpoint, self._delete)