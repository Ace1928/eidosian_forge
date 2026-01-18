import difflib
import numpy as np
import sys
from pathlib import Path
import pytest
import matplotlib as mpl
from matplotlib.testing import subprocess_run_for_testing
from matplotlib import pyplot as plt
def extract_documented_functions(lines):
    """
        Return a list of all the functions that are mentioned in the
        autosummary blocks contained in *lines*.

        An autosummary block looks like this::

            .. autosummary::
               :toctree: _as_gen
               :template: autosummary.rst
               :nosignatures:

               plot
               plot_date

        """
    functions = []
    in_autosummary = False
    for line in lines:
        if not in_autosummary:
            if line.startswith('.. autosummary::'):
                in_autosummary = True
        else:
            if not line or line.startswith('   :'):
                continue
            if not line[0].isspace():
                in_autosummary = False
                continue
            functions.append(line.strip())
    return functions