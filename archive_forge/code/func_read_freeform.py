import os
import re
import warnings
import numpy as np
from copy import deepcopy
import ase
from ase.parallel import paropen
from ase.spacegroup import Spacegroup
from ase.geometry.cell import cellpar_to_cell
from ase.constraints import FixAtoms, FixedPlane, FixedLine, FixCartesian
from ase.utils import atoms_to_spglib_cell
import ase.units
def read_freeform(fd):
    """
    Read a CASTEP freeform file (the basic format of .cell and .param files)
    and return keyword-value pairs as a dict (values are strings for single
    keywords and lists of strings for blocks).
    """
    from ase.calculators.castep import CastepInputFile
    inputobj = CastepInputFile(keyword_tolerance=2)
    filelines = fd.readlines()
    keyw = None
    read_block = False
    block_lines = None
    for i, l in enumerate(filelines):
        L = re.split('[#!;]', l, 1)[0].strip()
        if L == '':
            continue
        lsplit = re.split('\\s*[:=]*\\s+', L, 1)
        if read_block:
            if lsplit[0].lower() == '%endblock':
                if len(lsplit) == 1 or lsplit[1].lower() != keyw:
                    raise ValueError('Out of place end of block at line %i in freeform file' % i + 1)
                else:
                    read_block = False
                    inputobj.__setattr__(keyw, block_lines)
            else:
                block_lines += [L]
        else:
            read_block = lsplit[0].lower() == '%block'
            if read_block:
                if len(lsplit) == 1:
                    raise ValueError('Unrecognizable block at line %i in io freeform file' % i + 1)
                else:
                    keyw = lsplit[1].lower()
            else:
                keyw = lsplit[0].lower()
            if read_block:
                block_lines = []
            else:
                inputobj.__setattr__(keyw, ' '.join(lsplit[1:]))
    return inputobj.get_attr_dict(types=True)