import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
@property
def simulator_name(self):
    simulator_name, _ = self.simulator_model.get_simulator_name_and_version()
    return simulator_name