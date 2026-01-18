from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import scheduler_hints as sh
from heat.engine.resources import volume_base as vb
from heat.engine import support
from heat.engine import translation
def check_snapshot_complete(self, backup_id):
    backup = self.client().backups.get(backup_id)
    if backup.status == 'creating':
        return False
    if backup.status == 'available':
        return True
    raise exception.Error(backup.fail_reason)