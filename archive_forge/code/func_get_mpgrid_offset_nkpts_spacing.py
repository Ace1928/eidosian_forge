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
def get_mpgrid_offset_nkpts_spacing(self) -> tuple[tuple[int, int, int], Vector3D, int, float] | None:
    """
        Retrieves the MP grid, the grid offsets, number of kpoints, and maximum kpoint spacing.

        Returns:
            tuple[tuple[int, int, int], Vector3D, int, float]: (MP grid), (offsets), No. kpts, max spacing)
        """
    for rem in self._res.REMS:
        if rem.strip().startswith('MP grid'):
            srem = rem.split()
            p, q, r = map(int, srem[2:5])
            po, qo, ro = map(float, srem[6:9])
            return ((p, q, r), (po, qo, ro), int(srem[11]), float(srem[13]))
    self._raise_or_none(ResParseError('Could not find line with MP grid.'))
    return None