import collections
import copy
import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.heat import software_config as sc
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import support
@staticmethod
def get_input_key_arg(snippet, input_key):
    if len(snippet) != 1:
        return None
    fn_name, fn_arg = next(iter(snippet.items()))
    if fn_name == input_key and isinstance(fn_arg, str):
        return fn_arg