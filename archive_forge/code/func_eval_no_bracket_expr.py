import os
import operator as op
import re
import warnings
from collections import OrderedDict
from os import path
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from ase.calculators.calculator import kpts2ndarray, kpts2sizeandoffsets
from ase.dft.kpoints import kpoint_convert
from ase.constraints import FixAtoms, FixCartesian
from ase.data import chemical_symbols, atomic_numbers
from ase.units import create_units
from ase.utils import iofunction
def eval_no_bracket_expr(full_text):
    """Calculate value of a mathematical expression, no brackets."""
    exprs = [('+', op.add), ('*', op.mul), ('/', op.truediv), ('^', op.pow)]
    full_text = full_text.lstrip('(').rstrip(')')
    try:
        return float(full_text)
    except ValueError:
        for symbol, func in exprs:
            if symbol in full_text:
                left, right = full_text.split(symbol, 1)
                return func(eval_no_bracket_expr(left), eval_no_bracket_expr(right))