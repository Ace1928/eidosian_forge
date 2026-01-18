from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def aims_uuid(self) -> str:
    """The aims-uuid for the calculation."""
    line_start = self.reverse_search_for(['aims_uuid'])
    if line_start == LINE_NOT_FOUND:
        raise AimsParseError('This file does not appear to be an aims-output file')
    return self.lines[line_start].split(':')[1].strip()