import numpy as np
from .structreader import Unpacker
from .utils import find_private_section
def get_g_vector(csa_dict):
    return get_vector(csa_dict, 'DiffusionGradientDirection', 3)