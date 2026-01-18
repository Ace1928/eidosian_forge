from . import kimpy_wrappers
from .exceptions import KIMCalculatorError
from .calculators import (
def get_model_supported_species(model_name):
    if _is_portable_model(model_name):
        with kimpy_wrappers.PortableModel(model_name, debug=False) as pm:
            supported_species, _ = pm.get_model_supported_species_and_codes()
    else:
        with kimpy_wrappers.SimulatorModel(model_name) as sm:
            supported_species = sm.supported_species
    return supported_species