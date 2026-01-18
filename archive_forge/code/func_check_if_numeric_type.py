import logging
import sys
from pyomo.common.dependencies import numpy_available
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import TemplateExpressionError
def check_if_numeric_type(obj):
    """Test if the argument behaves like a numeric type.

    We check for "numeric types" by checking if we can add zero to it
    without changing the object's type, and that the object compares to
    0 in a meaningful way.  If that works, then we register the type in
    :py:attr:`native_numeric_types`.

    """
    obj_class = obj.__class__
    if obj_class in native_types:
        return obj_class in native_numeric_types
    if 'numpy' in obj_class.__module__:
        bool(numpy_available)
        if obj_class in native_types:
            return obj_class in native_numeric_types
    try:
        obj_plus_0 = obj + 0
        obj_p0_class = obj_plus_0.__class__
        if not (obj < 0) ^ (obj >= 0):
            return False
        hash(obj)
    except:
        return False
    if obj_p0_class is obj_class or obj_p0_class in native_numeric_types:
        RegisterNumericType(obj_class)
        logger.warning(f'Dynamically registering the following numeric type:\n    {obj_class.__module__}.{obj_class.__name__}\nDynamic registration is supported for convenience, but there are known\nlimitations to this approach.  We recommend explicitly registering\nnumeric types using RegisterNumericType() or RegisterIntegerType().')
        return True
    else:
        return False