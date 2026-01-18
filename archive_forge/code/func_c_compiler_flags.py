from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def c_compiler_flags(self) -> str | None:
    """The C compiler flags used to make FHI-aims."""
    line_start = self.reverse_search_for(['C compiler flags'])
    if line_start == LINE_NOT_FOUND:
        return None
    return self.lines[line_start].split(':')[1].strip()