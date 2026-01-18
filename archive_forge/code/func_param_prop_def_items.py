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
def param_prop_def_items(name, schema, template_type):
    if template_type == 'hot':
        param_def = cls._hot_param_def_from_prop(schema)
        prop_def = cls._hot_prop_def_from_prop(name, schema)
    else:
        param_def = cls._param_def_from_prop(schema)
        prop_def = cls._prop_def_from_prop(name, schema)
    return ((name, param_def), (name, prop_def))