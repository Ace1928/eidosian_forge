from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
@property
def roles(self):
    if self.system_scoped:
        roles = self._get_system_roles()
    elif self.trust_scoped:
        roles = self._get_trust_roles()
    elif self.oauth_scoped:
        roles = self._get_oauth_roles()
    elif self.is_federated and (not self.unscoped):
        roles = self._get_federated_roles()
    elif self.domain_scoped:
        roles = self._get_domain_roles()
    elif self.application_credential_id and self.project_id:
        roles = self._get_application_credential_roles()
    elif self.project_scoped:
        roles = self._get_project_roles()
    else:
        roles = []
    return roles