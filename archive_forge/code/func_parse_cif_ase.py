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
def parse_cif_ase(fileobj) -> Iterator[CIFBlock]:
    """Parse a CIF file using ase CIF parser."""
    if isinstance(fileobj, str):
        with open(fileobj, 'rb') as fileobj:
            data = fileobj.read()
    else:
        data = fileobj.read()
    if isinstance(data, bytes):
        data = data.decode('latin1')
    data = format_unicode(data)
    lines = [e for e in data.split('\n') if len(e) > 0]
    if len(lines) > 0 and lines[0].rstrip() == '#\\#CIF_2.0':
        warnings.warn('CIF v2.0 file format detected; `ase` CIF reader might incorrectly interpret some syntax constructions, use `pycodcif` reader instead')
    lines = [''] + lines[::-1]
    while lines:
        line = lines.pop().strip()
        if not line or line.startswith('#'):
            continue
        yield parse_block(lines, line)