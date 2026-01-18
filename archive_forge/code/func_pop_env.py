import inspect
from weakref import ref as weakref_ref
from pyomo.common.errors import PyomoException
from pyomo.common.deprecation import deprecated, deprecation_warning
@staticmethod
@deprecated('The PluginGlobals environment manager is deprecated: Pyomo only supports a single global environment', version='6.0')
def pop_env():
    pass