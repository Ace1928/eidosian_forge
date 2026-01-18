import math
import json
from jmespath import exceptions
from jmespath.compat import string_type as STRING_TYPE
from jmespath.compat import get_methods, with_metaclass
def keyfunc(x):
    result = expref.visit(expref.expression, x)
    actual_typename = type(result).__name__
    jmespath_type = self._convert_to_jmespath_type(actual_typename)
    if jmespath_type not in allowed_types:
        raise exceptions.JMESPathTypeError(function_name, result, jmespath_type, allowed_types)
    return result