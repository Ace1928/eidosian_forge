import pytest
import numpy as np
from ase.geometry.bravais_type_engine import generate_niggli_op_table
def ref_info_iter():
    for key, val in ref_info.items():
        yield (key, val)