import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
@property
def supported_units(self):
    try:
        supported_units = self.metadata['units'][0]
    except (KeyError, IndexError):
        raise KIMModelInitializationError('Unable to determine supported units of simulator model {}.'.format(self.model_name))
    return supported_units