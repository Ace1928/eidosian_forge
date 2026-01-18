import numpy as np
import pytest
from ase.io.formats import ioformats, match_magic
def lammpsdump_headers():
    actual_magic = 'ITEM: TIMESTEP'
    yield actual_magic
    yield f'anything\n{actual_magic}\nanything'