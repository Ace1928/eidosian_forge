import inspect
import sys
from .__wrapt__ import FunctionWrapper
def patch_function_wrapper(module, name, enabled=None):

    def _wrapper(wrapper):
        return wrap_object(module, name, FunctionWrapper, (wrapper, enabled))
    return _wrapper