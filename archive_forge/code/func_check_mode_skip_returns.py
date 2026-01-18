from __future__ import absolute_import, division, print_function
import traceback
from functools import wraps
from ansible_collections.community.general.plugins.module_utils.mh.exceptions import ModuleHelperException
def check_mode_skip_returns(callable=None, value=None):

    def deco(func):
        if callable is not None:

            @wraps(func)
            def wrapper_callable(self, *args, **kwargs):
                if self.module.check_mode:
                    return callable(self, *args, **kwargs)
                return func(self, *args, **kwargs)
            return wrapper_callable
        if value is not None:

            @wraps(func)
            def wrapper_value(self, *args, **kwargs):
                if self.module.check_mode:
                    return value
                return func(self, *args, **kwargs)
            return wrapper_value
    if callable is None and value is None:
        return check_mode_skip
    return deco