from __future__ import annotations
import collections
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
def names_and_values(self, key, minval=None, minfract=None, sorted=True):
    """
        Select the entries whose value[key] is >= minval or whose fraction[key] is >= minfract
        Return the names of the sections and the corresponding values.
        """
    values = self.get_values(key)
    names = self.get_values('name')
    new_names, new_values = ([], [])
    other_val = 0.0
    if minval is not None:
        assert minfract is None
        for name, val in zip(names, values):
            if val >= minval:
                new_names.append(name)
                new_values.append(val)
            else:
                other_val += val
        new_names.append(f'below minval {minval}')
        new_values.append(other_val)
    elif minfract is not None:
        assert minval is None
        total = self.sum_sections(key)
        for name, val in zip(names, values):
            if val / total >= minfract:
                new_names.append(name)
                new_values.append(val)
            else:
                other_val += val
        new_names.append(f'below minfract {minfract}')
        new_values.append(other_val)
    else:
        new_names, new_values = (names, values)
    if sorted:
        nandv = list(zip(new_names, new_values))
        nandv.sort(key=lambda t: t[1])
        new_names, new_values = ([n[0] for n in nandv], [n[1] for n in nandv])
    return (new_names, new_values)