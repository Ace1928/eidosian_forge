from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
@property
def audit_ids(self):
    if self.parent_audit_id:
        return [self.audit_id, self.parent_audit_id]
    return [self.audit_id]