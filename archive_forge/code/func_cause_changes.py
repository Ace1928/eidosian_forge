from __future__ import absolute_import, division, print_function
import traceback
from functools import wraps
from ansible_collections.community.general.plugins.module_utils.mh.exceptions import ModuleHelperException
def cause_changes(on_success=None, on_failure=None):

    def deco(func):
        if on_success is None and on_failure is None:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                self = args[0]
                func(*args, **kwargs)
                if on_success is not None:
                    self.changed = on_success
            except Exception:
                if on_failure is not None:
                    self.changed = on_failure
                raise
        return wrapper
    return deco