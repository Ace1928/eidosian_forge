from statsmodels.compat.python import lzip
import numpy as np
def normalize_cov_type(cov_type):
    """
    Normalize the cov_type string to a canonical version

    Parameters
    ----------
    cov_type : str

    Returns
    -------
    normalized_cov_type : str
    """
    if cov_type == 'nw-panel':
        cov_type = 'hac-panel'
    if cov_type == 'nw-groupsum':
        cov_type = 'hac-groupsum'
    return cov_type