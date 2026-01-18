import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
@property
def model_defn(self):
    return self.metadata['model-defn']