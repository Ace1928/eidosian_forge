import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
@check_call_wrapper
def get_callback_support_status(self, name):
    return self.compute_args.get_callback_support_status(name)