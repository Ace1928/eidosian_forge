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
def convert_vasp_outcar_stress(stress: Sequence):
    """Helper function to convert the stress line in an OUTCAR to the
    expected units in ASE """
    stress_arr = -np.array(stress)
    shape = stress_arr.shape
    if shape != (6,):
        raise ValueError('Stress has the wrong shape. Expected (6,), got {}'.format(shape))
    stress_arr = stress_arr[[0, 1, 2, 4, 5, 3]] * 0.1 * ase.units.GPa
    return stress_arr