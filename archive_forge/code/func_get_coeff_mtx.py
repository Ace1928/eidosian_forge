import numpy as np
from chempy.units import unit_of, to_unitless
def get_coeff_mtx(substances, stoichs):
    """
    Create a net stoichiometry matrix from reactions
    described by pairs of dictionaries.

    Parameters
    ----------
    substances : sequence of keys in stoichs dict pairs
    stoichs : sequence of pairs of dicts
        Pairs of reactant and product dicts mapping substance keys
        to stoichiometric coefficients (integers).

    Returns
    -------
    2 dimensional array of shape (len(substances), len(stoichs))

    """
    A = np.zeros((len(substances), len(stoichs)), dtype=int)
    for ri, sb in enumerate(substances):
        for ci, (reac, prod) in enumerate(stoichs):
            A[ri, ci] = prod.get(sb, 0) - reac.get(sb, 0)
    return A