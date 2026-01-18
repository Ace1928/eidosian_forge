from abc import ABC, abstractmethod
from typing import (Dict, Any, Sequence, TextIO, Iterator, Optional, Union,
import re
from warnings import warn
from pathlib import Path, PurePath
import numpy as np
import ase
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import ParseError, read
from ase.io.utils import ImageChunk
from ase.calculators.singlepoint import SinglePointDFTCalculator, SinglePointKPoint
def read_constraints_from_file(directory):
    directory = Path(directory)
    constraint = None
    for filename in ('CONTCAR', 'POSCAR'):
        if (directory / filename).is_file():
            constraint = read(directory / filename, format='vasp').constraints
            break
    return constraint