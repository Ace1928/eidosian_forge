import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
def get_monkhorst_shape(kpts):
    warnings.warn('Use get_monkhorst_pack_size_and_offset()[0] instead.')
    return get_monkhorst_pack_size_and_offset(kpts)[0]