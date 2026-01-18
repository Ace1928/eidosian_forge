import collections
import enum
import logging
import math
import operator
from pyomo.common.dependencies import attempt_import
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_common import (
from pyomo.core.expr.base import ExpressionBase, NPV_Mixin, visitor
@deprecated('The implicit recasting of a "not potentially variable" expression node to a potentially variable one is no longer supported (this violates that immutability promise for Pyomo5 expression trees).', version='6.4.3')
def create_potentially_variable_object(self):
    """
        Create a potentially variable version of this object.

        This method returns an object that is a potentially variable
        version of the current object.  In the simplest
        case, this simply sets the value of `__class__`:

            self.__class__ = self.__class__.__mro__[1]

        Note that this method is allowed to modify the current object
        and return it.  But in some cases it may create a new
        potentially variable object.

        Returns:
            An object that is potentially variable.
        """
    if not self.is_potentially_variable():
        logger.error('recasting a non-potentially variable expression to a potentially variable one violates the immutability promise for Pyomo expression trees.')
        self.__class__ = self.potentially_variable_base_class()
    return self