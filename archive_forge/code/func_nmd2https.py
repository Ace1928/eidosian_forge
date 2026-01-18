import json
import numpy as np
import ase.units as units
from ase import Atoms
from ase.data import chemical_symbols
def nmd2https(uri):
    """Get https URI corresponding to given nmd:// URI."""
    assert uri.startswith('nmd://')
    return nomad_api_template.format(hash=uri[6:])