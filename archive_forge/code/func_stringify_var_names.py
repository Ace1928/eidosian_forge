import itertools
import os
import re
import numpy as np
def stringify_var_names(var_list, delimiter=''):
    """

    Parameters
    ----------
    var_list : list[str]
        Each list element is the name of a variable.

    Returns
    -------
    result : str
        Concatenated variable names.
    """
    result = var_list[0]
    for var_name in var_list[1:]:
        result += delimiter + var_name
    return result.lower()