from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants
import scipy.special
from monty.json import MSONable
from tqdm import tqdm
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Vasprun, Waveder
def step_methfessel_paxton(x, n):
    """
    S_n (x) = (1 + erf x)/2 - exp -x^2 * sum_i=1^n A_i H_{2i-1}(x)
    where H is a Hermite polynomial and
    A_i = (-1)^i / ( i! 4^i sqrt(pi) ).
    """
    ii = np.arange(1, n + 1)
    A = (-1) ** ii / (scipy.special.factorial(ii) * 4 ** ii * np.sqrt(np.pi))
    H = scipy.special.eval_hermite(ii * 2 - 1, np.tile(x, (len(ii), 1)).T)
    return (1.0 + scipy.special.erf(x)) / 2.0 - np.exp(-(x * x)) * np.dot(A, H.T)