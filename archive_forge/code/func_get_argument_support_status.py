import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
@check_call_wrapper
def get_argument_support_status(self, name):
    return self.compute_args.get_argument_support_status(name)