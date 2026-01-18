from __future__ import annotations
import os
import re
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
def append_transformation(self, transformation, extend_collection=False, clear_redo=True):
    """Appends a transformation to all TransformedStructures.

        Args:
            transformation: Transformation to append
            extend_collection: Whether to use more than one output structure
                from one-to-many transformations. extend_collection can be a
                number, which determines the maximum branching for each transformation.
            clear_redo (bool): Whether to clear the redo list. By default,
                this is True, meaning any appends clears the history of
                undoing. However, when using append_transformation to do a
                redo, the redo list should not be cleared to allow multiple redos.

        Returns:
            list[bool]: corresponding to initial transformed structures each boolean
                describes whether the transformation altered the structure
        """
    if self.ncores and transformation.use_multiprocessing:
        with Pool(self.ncores) as p:
            z = ((x, transformation, extend_collection, clear_redo) for x in self.transformed_structures)
            trafo_new_structs = p.map(_apply_transformation, z, 1)
            self.transformed_structures = []
            for ts in trafo_new_structs:
                self.transformed_structures.extend(ts)
    else:
        new_structures = []
        for x in self.transformed_structures:
            new = x.append_transformation(transformation, extend_collection, clear_redo=clear_redo)
            if new is not None:
                new_structures.extend(new)
        self.transformed_structures.extend(new_structures)