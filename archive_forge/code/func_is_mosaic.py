import numpy as np
from .structreader import Unpacker
from .utils import find_private_section
def is_mosaic(csa_dict):
    """Return True if the data is of Mosaic type

    Parameters
    ----------
    csa_dict : dict
       dict containing read CSA data

    Returns
    -------
    tf : bool
       True if the `dcm_data` appears to be of Siemens mosaic type,
       False otherwise
    """
    if csa_dict is None:
        return False
    if get_acq_mat_txt(csa_dict) is None:
        return False
    n_o_m = get_n_mosaic(csa_dict)
    return not n_o_m is None and n_o_m != 0