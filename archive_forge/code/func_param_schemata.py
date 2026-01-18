import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import template as cfn_template
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters
from heat.engine import rsrc_defn
from heat.engine import template_common
def param_schemata(self, param_defaults=None):
    parameter_section = self.t.get(self.PARAMETERS) or {}
    pdefaults = param_defaults or {}
    for name, schema in parameter_section.items():
        if name in pdefaults:
            parameter_section[name]['default'] = pdefaults[name]
    params = parameter_section.items()
    return dict(((name, self.param_schema_class.from_dict(name, schema)) for name, schema in params))