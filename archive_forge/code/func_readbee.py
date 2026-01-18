import os
from typing import Any, Union
import numpy as np
from ase import Atoms
from ase.io.jsonio import read_json, write_json
from ase.parallel import world, parprint
def readbee(fname: str, all: bool=False):
    if not fname.endswith('.bee'):
        fname += '.bee'
    with open(fname, 'r') as fd:
        e, de, contribs, seed, xc = read_json(fd, always_array=False)
    if all:
        return (e, de, contribs, seed, xc)
    else:
        return e + de