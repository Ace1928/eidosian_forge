import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import (
def config_prover9(self, binary_location, verbose=False):
    if binary_location is None:
        self._binary_location = None
        self._prover9_bin = None
    else:
        name = 'prover9'
        self._prover9_bin = nltk.internals.find_binary(name, path_to_bin=binary_location, env_vars=['PROVER9'], url='https://www.cs.unm.edu/~mccune/prover9/', binary_names=[name, name + '.exe'], verbose=verbose)
        self._binary_location = self._prover9_bin.rsplit(os.path.sep, 1)