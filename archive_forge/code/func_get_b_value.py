import numpy as np
from .structreader import Unpacker
from .utils import find_private_section
def get_b_value(csa_dict):
    return get_scalar(csa_dict, 'B_value')