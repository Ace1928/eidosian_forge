from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .core import cbc_path, pulp_cbc_path, coinMP_path, devnull, operating_system
import os
from .. import constants
from tempfile import mktemp
import ctypes
import warnings
def readsol_MPS(self, filename, lp, vs, variablesNames, constraintsNames, objectiveName=None):
    """
        Read a CBC solution file generated from an mps or lp file (possible different names)
        """
    values = {v.name: 0 for v in vs}
    reverseVn = {v: k for k, v in variablesNames.items()}
    reverseCn = {v: k for k, v in constraintsNames.items()}
    reducedCosts = {}
    shadowPrices = {}
    slacks = {}
    status, sol_status = self.get_status(filename)
    with open(filename) as f:
        for l in f:
            if len(l) <= 2:
                break
            l = l.split()
            if l[0] == '**':
                l = l[1:]
            vn = l[1]
            val = l[2]
            dj = l[3]
            if vn in reverseVn:
                values[reverseVn[vn]] = float(val)
                reducedCosts[reverseVn[vn]] = float(dj)
            if vn in reverseCn:
                slacks[reverseCn[vn]] = float(val)
                shadowPrices[reverseCn[vn]] = float(dj)
    return (status, values, reducedCosts, shadowPrices, slacks, sol_status)