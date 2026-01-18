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
def autolabel(symbols: Sequence[str]) -> List[str]:
    no: Dict[str, int] = {}
    labels = []
    for symbol in symbols:
        if symbol in no:
            no[symbol] += 1
        else:
            no[symbol] = 1
        labels.append('%s%d' % (symbol, no[symbol]))
    return labels