import numpy as np
from ase import Atoms
@classmethod
def initialize_individual(cls, parent, indi=None):
    """Initializes a new individual that inherits some parameters
        from the parent, and initializes the info dictionary.
        If the new individual already has more structure it can be
        supplied in the parameter indi."""
    if indi is None:
        indi = Atoms(pbc=parent.get_pbc(), cell=parent.get_cell())
    else:
        indi = indi.copy()
    indi.info['key_value_pairs'] = {'extinct': 0}
    indi.info['data'] = {}
    return indi