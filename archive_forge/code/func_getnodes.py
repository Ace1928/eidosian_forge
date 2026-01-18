from statsmodels.compat.python import lrange
from pprint import pprint
import numpy as np
def getnodes(tree):
    """
    walk tree to get list of branches and list of leaves

    Parameters
    ----------
    tree : list of tuples
        tree as defined for RU2NMNL

    Returns
    -------
    branch : list
        list of all branch names
    leaves : list
        list of all leaves names

    """
    if isinstance(tree, tuple):
        name, subtree = tree
        ab = [name]
        al = []
        if len(subtree) == 1:
            adeg = [name]
        else:
            adeg = []
        for st in subtree:
            b, l, d = getnodes(st)
            ab.extend(b)
            al.extend(l)
            adeg.extend(d)
        return (ab, al, adeg)
    return ([], [tree], [])