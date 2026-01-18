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
def needs_replace_failed(self):
    if not self.resource_id:
        return True
    with self.client_plugin().ignore_not_found:
        vol = self.client().volumes.get(self.resource_id)
        return vol.status in ('error', 'deleting')
    return True