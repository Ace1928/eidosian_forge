from statsmodels.compat.python import lrange
from pprint import pprint
import numpy as np
def getbranches(tree):
    """
    walk tree to get list of branches

    Parameters
    ----------
    tree : list of tuples
        tree as defined for RU2NMNL

    Returns
    -------
    branch : list
        list of all branch names

    """
    if isinstance(tree, tuple):
        name, subtree = tree
        a = [name]
        for st in subtree:
            a.extend(getbranches(st))
        return a
    return []