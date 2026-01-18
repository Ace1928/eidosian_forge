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
def parse_cif_loop_headers(lines: List[str]) -> Iterator[str]:
    header_pattern = '\\s*(_\\S*)'
    while lines:
        line = lines.pop()
        match = re.match(header_pattern, line)
        if match:
            header = match.group(1).lower()
            yield header
        elif re.match('\\s*#', line):
            continue
        else:
            lines.append(line)
            return