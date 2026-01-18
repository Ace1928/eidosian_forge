from keystone.common import cache
from keystone.common import manager
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.models import revoke_model
from keystone import notifications
def revoke_by_audit_chain_id(self, audit_chain_id, project_id=None, domain_id=None):
    self._assert_not_domain_and_project_scoped(domain_id=domain_id, project_id=project_id)
    self.revoke(revoke_model.RevokeEvent(audit_chain_id=audit_chain_id, domain_id=domain_id, project_id=project_id))