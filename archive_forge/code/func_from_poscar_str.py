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
@classmethod
def from_poscar_str(cls, poscar_string: str, transformations: list[AbstractTransformation] | None=None) -> Self:
    """Generates TransformedStructure from a poscar string.

        Args:
            poscar_string (str): Input POSCAR string.
            transformations (list[Transformation]): Sequence of transformations
                to be applied to the input structure.
        """
    poscar = Poscar.from_str(poscar_string)
    if not poscar.true_names:
        raise ValueError('Transformation can be created only from POSCAR strings with proper VASP5 element symbols.')
    raw_string = re.sub("'", '"', poscar_string)
    struct = poscar.structure
    source_info = {'source': 'POSCAR', 'datetime': str(datetime.datetime.now()), 'original_file': raw_string}
    return cls(struct, transformations, history=[source_info])