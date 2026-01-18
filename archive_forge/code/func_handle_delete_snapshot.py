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
def handle_delete_snapshot(self, snapshot):
    backup_id = snapshot['resource_data'].get('backup_id')
    if not backup_id:
        return
    try:
        self.client().backups.delete(backup_id)
    except Exception as ex:
        self.client_plugin().ignore_not_found(ex)
        return
    else:
        return backup_id