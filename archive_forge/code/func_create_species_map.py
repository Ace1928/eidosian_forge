import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import compare_atoms
from . import kimpy_wrappers
from . import neighborlist
def create_species_map(self):
    """Get all the supported species of the KIM model and the
        corresponding integer codes used by the model

        Returns
        -------
        species_map : dict
            key : str
                chemical symbols (e.g. "Ar")
            value : int
                species integer code (e.g. 1)
        """
    supported_species, codes = self.get_model_supported_species_and_codes()
    species_map = dict()
    for i, spec in enumerate(supported_species):
        species_map[spec] = codes[i]
        if self.debug:
            print('Species {} is supported and its code is: {}'.format(spec, codes[i]))
    return species_map