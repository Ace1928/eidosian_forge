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
def format_cell(cell: Cell) -> str:
    assert cell.rank == 3
    lines = []
    for name, value in zip(CIFBlock.cell_tags, cell.cellpar()):
        line = '{:20} {:g}\n'.format(name, value)
        lines.append(line)
    assert len(lines) == 6
    return ''.join(lines)