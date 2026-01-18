import collections
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
from heat.engine import function
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import support
from heat.engine import translation as trans
def get_user_value(self, key):
    if key not in self:
        raise KeyError(_('Invalid Property %s') % key)
    prop = self.props[key]
    value, found = self._resolve_user_value(key, prop, validate=False)
    return value