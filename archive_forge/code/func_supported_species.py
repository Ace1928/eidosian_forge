import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
@property
def supported_species(self):
    supported_species = []
    for spec_code in range(self.num_supported_species):
        species = check_call(self.simulator_model.get_supported_species, spec_code)
        supported_species.append(species)
    return tuple(supported_species)