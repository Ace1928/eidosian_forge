import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def get_standard_key(key):
    """
    Standard ASE parameter format is to USE unerbar(_) instead of dot(.). Also,
    It is recommended to use lower case alphabet letter. Not Upper. Thus, we
    change the key to standard key
    For example:
        'scf.XcType' -> 'scf_xctype'
    """
    if isinstance(key, str):
        return key.lower().replace('.', '_')
    elif isinstance(key, list):
        return [k.lower().replace('.', '_') for k in key]
    else:
        return [k.lower().replace('.', '_') for k in key]