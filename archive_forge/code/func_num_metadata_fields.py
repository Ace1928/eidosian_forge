import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
@property
def num_metadata_fields(self):
    return self.simulator_model.get_number_of_simulator_fields()