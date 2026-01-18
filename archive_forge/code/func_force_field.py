from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from string import Template
from monty.io import zopen
from pymatgen.core import Structure
from pymatgen.io.core import InputGenerator
from pymatgen.io.lammps.data import CombinedData, LammpsData
from pymatgen.io.lammps.inputs import LammpsInputFile
from pymatgen.io.lammps.sets import LammpsInputSet
@property
def force_field(self) -> str:
    """Return the details of the force field commands passed to the generator."""
    return self.settings['force_field']