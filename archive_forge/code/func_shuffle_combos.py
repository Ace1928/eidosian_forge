from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
def shuffle_combos(self, shuffle_dic, internal_type):
    dfns = []
    for dfn in internal_type:
        append = True
        all_indices = [idx[0:-1] for idx in dfn[1]]
        new_dfn = [dfn[0], list(dfn[1])]
        for i, indices in enumerate(all_indices):
            for old in indices:
                if old in shuffle_dic:
                    new_dfn[1][i][indices.index(old)] = shuffle_dic[old]
                else:
                    append = False
                    break
            if not append:
                break
        if append:
            dfns.append(new_dfn)
    return dfns