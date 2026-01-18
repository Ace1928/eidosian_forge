import re
from pyomo.common.deprecation import deprecated
from pyomo.core.base.componentuid import ComponentUID
@deprecated("The 'remove_obj' method is no longer necessary now that 'getname' does not support the use of a name buffer", version='6.4.1')
def remove_obj(self, obj):
    pass