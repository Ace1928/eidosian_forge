import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
@property
def num_supported_species(self):
    num_supported_species = self.simulator_model.get_number_of_supported_species()
    if num_supported_species == 0:
        raise KIMModelInitializationError('Unable to determine supported species of simulator model {}.'.format(self.model_name))
    else:
        return num_supported_species