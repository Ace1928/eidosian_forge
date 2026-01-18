from __future__ import annotations
import os
import re
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
def redo_next_change(self) -> None:
    """Redo the last undone transformation in the TransformedStructure.

        Raises:
            IndexError if already at the latest change.
        """
    for x in self.transformed_structures:
        x.redo_next_change()