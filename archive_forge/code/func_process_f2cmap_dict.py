import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def process_f2cmap_dict(f2cmap_all, new_map, c2py_map, verbose=False):
    """
    Update the Fortran-to-C type mapping dictionary with new mappings and
    return a list of successfully mapped C types.

    This function integrates a new mapping dictionary into an existing
    Fortran-to-C type mapping dictionary. It ensures that all keys are in
    lowercase and validates new entries against a given C-to-Python mapping
    dictionary. Redefinitions and invalid entries are reported with a warning.

    Parameters
    ----------
    f2cmap_all : dict
        The existing Fortran-to-C type mapping dictionary that will be updated.
        It should be a dictionary of dictionaries where the main keys represent
        Fortran types and the nested dictionaries map Fortran type specifiers
        to corresponding C types.

    new_map : dict
        A dictionary containing new type mappings to be added to `f2cmap_all`.
        The structure should be similar to `f2cmap_all`, with keys representing
        Fortran types and values being dictionaries of type specifiers and their
        C type equivalents.

    c2py_map : dict
        A dictionary used for validating the C types in `new_map`. It maps C
        types to corresponding Python types and is used to ensure that the C
        types specified in `new_map` are valid.

    verbose : boolean
        A flag used to provide information about the types mapped

    Returns
    -------
    tuple of (dict, list)
        The updated Fortran-to-C type mapping dictionary and a list of
        successfully mapped C types.
    """
    f2cmap_mapped = []
    new_map_lower = {}
    for k, d1 in new_map.items():
        d1_lower = {k1.lower(): v1 for k1, v1 in d1.items()}
        new_map_lower[k.lower()] = d1_lower
    for k, d1 in new_map_lower.items():
        if k not in f2cmap_all:
            f2cmap_all[k] = {}
        for k1, v1 in d1.items():
            if v1 in c2py_map:
                if k1 in f2cmap_all[k]:
                    outmess("\tWarning: redefinition of {'%s':{'%s':'%s'->'%s'}}\n" % (k, k1, f2cmap_all[k][k1], v1))
                f2cmap_all[k][k1] = v1
                if verbose:
                    outmess('\tMapping "%s(kind=%s)" to "%s"\n' % (k, k1, v1))
                f2cmap_mapped.append(v1)
            elif verbose:
                errmess("\tIgnoring map {'%s':{'%s':'%s'}}: '%s' must be in %s\n" % (k, k1, v1, v1, list(c2py_map.keys())))
    return (f2cmap_all, f2cmap_mapped)