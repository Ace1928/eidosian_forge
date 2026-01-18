from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
@property
def dbinstance(self):
    """Get the trove dbinstance."""
    if not self._dbinstance and self.resource_id:
        self._dbinstance = self.client().instances.get(self.resource_id)
    return self._dbinstance