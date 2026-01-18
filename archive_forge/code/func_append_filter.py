from __future__ import annotations
import datetime
import json
import re
from typing import TYPE_CHECKING, Any
from warnings import warn
from monty.json import MSONable, jsanitize
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
from pymatgen.transformations.transformation_abc import AbstractTransformation
from pymatgen.util.provenance import StructureNL
def append_filter(self, structure_filter: AbstractStructureFilter) -> None:
    """Adds a filter.

        Args:
            structure_filter (StructureFilter): A filter implementing the
                AbstractStructureFilter API. Tells transmuter what structures to retain.
        """
    h_dict = structure_filter.as_dict()
    h_dict['input_structure'] = self.final_structure.as_dict()
    self.history.append(h_dict)