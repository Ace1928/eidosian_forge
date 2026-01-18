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
def get_airss_version(self) -> tuple[str, date] | None:
    """
        Retrieves the version of AIRSS that was used along with the build date (not compile date).

        Returns:
            tuple[str, date] (version string, date)
        """
    for rem in self._res.REMS:
        if rem.strip().startswith('AIRSS Version'):
            date = self._parse_date(rem)
            v = rem.split()[2]
            return (v, date)
    self._raise_or_none(ResParseError('Could not find line with AIRSS version.'))
    return None