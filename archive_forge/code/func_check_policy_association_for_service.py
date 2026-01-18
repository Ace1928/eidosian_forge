from keystoneclient import base
from keystoneclient.i18n import _
from keystoneclient.v3 import endpoints
from keystoneclient.v3 import policies
def check_policy_association_for_service(self, policy, service):
    """Check an association between a policy and a service."""
    return self._act_on_policy_association_for_service(policy, service, self._head)