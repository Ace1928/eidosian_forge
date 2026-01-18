import io
import re
import shlex
import warnings
from typing import Dict, List, Tuple, Optional, Union, Iterator, Any, Sequence
import collections.abc
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.spacegroup import crystal
from ase.spacegroup.spacegroup import spacegroup_from_data, Spacegroup
from ase.io.cif_unicode import format_unicode, handle_subscripts
from ase.utils import iofunction
def expand_kinds(atoms, coords):
    symbols = list(atoms.symbols)
    coords = list(coords)
    occupancies = [1] * len(symbols)
    occ_info = atoms.info.get('occupancy')
    kinds = atoms.arrays.get('spacegroup_kinds')
    if occ_info is not None and kinds is not None:
        for i, kind in enumerate(kinds):
            occ_info_kind = occ_info[str(kind)]
            symbol = symbols[i]
            if symbol not in occ_info_kind:
                raise BadOccupancies('Occupancies present but no occupancy info for "{symbol}"')
            occupancies[i] = occ_info_kind[symbol]
            for sym, occ in occ_info[str(kind)].items():
                if sym != symbols[i]:
                    symbols.append(sym)
                    coords.append(coords[i])
                    occupancies.append(occ)
    return (symbols, coords, occupancies)