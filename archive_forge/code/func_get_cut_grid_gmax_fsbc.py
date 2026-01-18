from and back to a string/file is not guaranteed to be reversible, i.e. a diff on the output
from __future__ import annotations
import datetime
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element, Lattice, PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.core import ParseError
def get_cut_grid_gmax_fsbc(self) -> tuple[float, float, float, str] | None:
    """
        Retrieves the cut-off energy, grid scale, Gmax, and finite basis set correction setting
        from the REM entries.

        Returns:
            tuple[float, float, float, str]: (cut-off, grid scale, Gmax, fsbc)
        """
    for rem in self._res.REMS:
        if rem.strip().startswith('Cut-off'):
            srem = rem.split()
            return (float(srem[1]), float(srem[5]), float(srem[7]), srem[10])
    self._raise_or_none(ResParseError('Could not find line with cut-off energy.'))
    return None