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
def parse_loop(lines: List[str]) -> Dict[str, List[CIFDataValue]]:
    """Parse a CIF loop. Returns a dict with column tag names as keys
    and a lists of the column content as values."""
    headers = list(parse_cif_loop_headers(lines))
    columns = parse_cif_loop_data(lines, len(headers))
    columns_dict = {}
    for i, header in enumerate(headers):
        if header in columns_dict:
            warnings.warn('Duplicated loop tags: {0}'.format(header))
        else:
            columns_dict[header] = columns[i]
    return columns_dict