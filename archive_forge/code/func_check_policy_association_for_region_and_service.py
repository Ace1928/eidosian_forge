from keystoneclient import base
from keystoneclient.i18n import _
from keystoneclient.v3 import endpoints
from keystoneclient.v3 import policies
def check_policy_association_for_region_and_service(self, policy, region, service):
    """Check an association between a policy and a service in a region."""
    return self._act_on_policy_association_for_region_and_service(policy, region, service, self._head)