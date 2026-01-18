import re
import numpy as np
from ase import Atoms
from ase.utils import reader, writer
from ase.io.utils import ImageIterator
from ase.io import ParseError
from .vasp_parsers import vasp_outcar_parsers as vop
from pathlib import Path
def atomtypes_outpot(posfname, numsyms):
    """Try to retrieve chemical symbols from OUTCAR or POTCAR

    If getting atomtypes from the first line in POSCAR/CONTCAR fails, it might
    be possible to find the data in OUTCAR or POTCAR, if these files exist.

    posfname -- The filename of the POSCAR/CONTCAR file we're trying to read

    numsyms -- The number of symbols we must find

    """
    posfpath = Path(posfname)
    fnames = [posfpath.with_name('POTCAR'), posfpath.with_name('OUTCAR')]
    fsc = []
    for fnpath in fnames:
        fsc.append(fnpath.parent / (fnpath.name + '.gz'))
        fsc.append(fnpath.parent / (fnpath.name + '.bz2'))
    for f in fsc:
        fnames.append(f)
    tried = []
    for fn in fnames:
        if fn in posfpath.parent.iterdir():
            tried.append(fn)
            at = get_atomtypes(fn)
            if len(at) == numsyms:
                return at
    raise ParseError('Could not determine chemical symbols. Tried files ' + str(tried))