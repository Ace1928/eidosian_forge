import os
import sys
import re
import numpy as np
import subprocess
from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from typing import Dict, Any
from xml.etree import ElementTree
import ase
from ase.io import read, jsonio
from ase.utils import PurePath
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.vasp.create_input import GenerateVaspInput
def read_sort(self):
    """Create the sorting and resorting list from ase-sort.dat.
        If the ase-sort.dat file does not exist, the sorting is redone.
        """
    sortfile = self._indir('ase-sort.dat')
    if os.path.isfile(sortfile):
        self.sort = []
        self.resort = []
        with open(sortfile, 'r') as fd:
            for line in fd:
                sort, resort = line.split()
                self.sort.append(int(sort))
                self.resort.append(int(resort))
    else:
        atoms = read(self._indir('CONTCAR'))
        self.initialize(atoms)