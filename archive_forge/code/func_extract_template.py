from functools import wraps
import inspect
import operator
from numba.core.extending import overload
from numba.core.types import ClassInstanceType
def extract_template(template, name):
    """
    Extract a code-generated function from a string template.
    """
    namespace = {}
    exec(template, namespace)
    return namespace[name]