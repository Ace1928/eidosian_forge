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
def get_pspots(self) -> dict[str, str]:
    """
        Retrieves the OTFG pseudopotential string that can be used to generate the
        pseudopotentials used in the calculation.

        Returns:
            dict[specie, potential]
        """
    pseudo_pots: dict[str, str] = {}
    for rem in self._res.REMS:
        srem = rem.split()
        if len(srem) == 2 and Element.is_valid_symbol(srem[0]):
            k, v = srem
            pseudo_pots[k] = v
    return pseudo_pots