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
def get_func_rel_disp(self) -> tuple[str, str, str] | None:
    """
        Retrieves the functional, relativity scheme, and dispersion correction from the REM entries.

        Returns:
            tuple[str, str, str]: (functional, relativity, dispersion)
        """
    for rem in self._res.REMS:
        if rem.strip().startswith('Functional'):
            srem = rem.split()
            return (' '.join(srem[1:4]), srem[5], srem[7])
    self._raise_or_none(ResParseError('Could not find functional, relativity, and dispersion.'))
    return None