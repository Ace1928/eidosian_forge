from __future__ import annotations
import os
import re
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
def append_transformed_structures(self, trafo_structs_or_transmuter):
    """Method is overloaded to accept either a list of transformed structures
        or transmuter, it which case it appends the second transmuter"s
        structures.

        Args:
            trafo_structs_or_transmuter: A list of transformed structures or a
                transmuter.
        """
    if isinstance(trafo_structs_or_transmuter, self.__class__):
        self.transformed_structures.extend(trafo_structs_or_transmuter.transformed_structures)
    else:
        for ts in trafo_structs_or_transmuter:
            assert isinstance(ts, TransformedStructure)
        self.transformed_structures.extend(trafo_structs_or_transmuter)