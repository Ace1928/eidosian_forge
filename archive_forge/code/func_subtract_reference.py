import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.calculators.calculator import PropertyNotImplementedError
def subtract_reference(self) -> 'BandStructure':
    """Return new band structure with reference energy subtracted."""
    return BandStructure(self.path, self.energies - self.reference, reference=0.0)