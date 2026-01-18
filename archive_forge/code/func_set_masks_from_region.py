import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import IOContext
from ase.geometry import get_distances
from ase.cell import Cell
def set_masks_from_region(self, region):
    """
        Sets masks from provided region array
        """
    self.qm_selection_mask = region == 'QM'
    buffer_mask = region == 'buffer'
    self.qm_buffer_mask = self.qm_selection_mask ^ buffer_mask