from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
def raise_invalid_exception(zone_type, prp):
    if self.properties.get(self.TYPE) == zone_type:
        if not self.properties.get(prp):
            msg = _('Property %(prp)s is required for zone type %(zone_type)s') % {'prp': prp, 'zone_type': zone_type}
            raise exception.StackValidationFailed(message=msg)