from __future__ import annotations
import multiprocessing as multiproc
import warnings
from string import ascii_uppercase
from time import time
from typing import TYPE_CHECKING
from pymatgen.command_line.mcsqs_caller import Sqs
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
def monte_carlo_sqs_structures(self) -> list:
    """Run `self.instances` Monte Carlo SQS search with Icet."""
    with multiproc.Pool(self.instances) as pool:
        return pool.starmap(self._single_monte_carlo_sqs_run, [() for _ in range(self.instances)])