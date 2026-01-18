from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def fortran_compiler(self) -> str | None:
    """The fortran compiler used to make FHI-aims."""
    line_start = self.reverse_search_for(['Fortran compiler      :'])
    if line_start == LINE_NOT_FOUND:
        raise AimsParseError('This file does not appear to be an aims-output file')
    return self.lines[line_start].split(':')[1].split('/')[-1].strip()