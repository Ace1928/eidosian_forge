import gzip
import struct
from collections import deque
from os.path import splitext
import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammps import convert
from ase.calculators.singlepoint import SinglePointCalculator
from ase.parallel import paropen
from ase.quaternions import Quaternions
def read_lammps_dump(infileobj, **kwargs):
    """Method which reads a LAMMPS dump file.

       LAMMPS chooses output method depending on the given suffix:
        - .bin  : binary file
        - .gz   : output piped through gzip
        - .mpiio: using mpiio (should be like cleartext,
                  with different ordering)
        - else  : normal clear-text format

    :param infileobj: string to file, opened file or file-like stream

    """
    opened = False
    if isinstance(infileobj, str):
        opened = True
        suffix = splitext(infileobj)[-1]
        if suffix == '.bin':
            fileobj = paropen(infileobj, 'rb')
        elif suffix == '.gz':
            fileobj = gzip.open(infileobj, 'rb')
        else:
            fileobj = paropen(infileobj)
    else:
        suffix = splitext(infileobj.name)[-1]
        fileobj = infileobj
    if suffix == '.bin':
        out = read_lammps_dump_binary(fileobj, **kwargs)
        if opened:
            fileobj.close()
        return out
    out = read_lammps_dump_text(fileobj, **kwargs)
    if opened:
        fileobj.close()
    return out