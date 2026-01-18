import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
@check_call_wrapper
def get_influence_distance(self):
    return self.kim_model.get_influence_distance()