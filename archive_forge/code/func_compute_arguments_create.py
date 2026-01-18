import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
def compute_arguments_create(self):
    return ComputeArguments(self, self.debug)