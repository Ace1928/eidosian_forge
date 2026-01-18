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
def get_section_name(self, section):
    cfn_to_hot_attrs = dict(zip(self._HOT_TO_CFN_ATTRS.values(), self._HOT_TO_CFN_ATTRS.keys()))
    return cfn_to_hot_attrs.get(section, section)