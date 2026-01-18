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
def read_cif(fileobj, index, store_tags=False, primitive_cell=False, subtrans_included=True, fractional_occupancies=True, reader='ase') -> Iterator[Atoms]:
    """Read Atoms object from CIF file. *index* specifies the data
    block number or name (if string) to return.

    If *index* is None or a slice object, a list of atoms objects will
    be returned. In the case of *index* is *None* or *slice(None)*,
    only blocks with valid crystal data will be included.

    If *store_tags* is true, the *info* attribute of the returned
    Atoms object will be populated with all tags in the corresponding
    cif data block.

    If *primitive_cell* is true, the primitive cell will be built instead
    of the conventional cell.

    If *subtrans_included* is true, sublattice translations are
    assumed to be included among the symmetry operations listed in the
    CIF file (seems to be the common behaviour of CIF files).
    Otherwise the sublattice translations are determined from setting
    1 of the extracted space group.  A result of setting this flag to
    true, is that it will not be possible to determine the primitive
    cell.

    If *fractional_occupancies* is true, the resulting atoms object will be
    tagged equipped with a dictionary `occupancy`. The keys of this dictionary
    will be integers converted to strings. The conversion to string is done
    in order to avoid troubles with JSON encoding/decoding of the dictionaries
    with non-string keys.
    Also, in case of mixed occupancies, the atom's chemical symbol will be
    that of the most dominant species.

    String *reader* is used to select CIF reader. Value `ase` selects
    built-in CIF reader (default), while `pycodcif` selects CIF reader based
    on `pycodcif` package.
    """
    images = []
    for block in parse_cif(fileobj, reader):
        if not block.has_structure():
            continue
        atoms = block.get_atoms(store_tags, primitive_cell, subtrans_included, fractional_occupancies=fractional_occupancies)
        images.append(atoms)
    for atoms in images[index]:
        yield atoms