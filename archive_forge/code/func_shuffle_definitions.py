from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
def shuffle_definitions(self, shuffle_dic, internal_type):
    dfns = []
    for dfn in internal_type:
        append = True
        new_dfn = [dfn[0], list(dfn[1])]
        for old in dfn[1]:
            if old in shuffle_dic:
                new_dfn[1][dfn[1].index(old)] = shuffle_dic[old]
            else:
                append = False
                break
        if append:
            dfns.append(new_dfn)
    return dfns