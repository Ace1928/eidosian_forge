from __future__ import annotations
import copy
import logging
import math
import os
import re
import struct
import warnings
from typing import TYPE_CHECKING, Any
import networkx as nx
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core import Molecule
from pymatgen.io.qchem.utils import (
def gradient_parser(filename: str='131.0') -> NDArray:
    """
    Parse the gradient data from a gradient scratch file.

    Args:
        filename: Path to the gradient scratch file. Defaults to "131.0".

    Returns:
        NDArray: The gradient, in units of Hartree/Bohr.
    """
    tmp_grad_data: list[float] = []
    with zopen(filename, mode='rb') as file:
        binary = file.read()
    tmp_grad_data.extend((struct.unpack('d', binary[ii * 8:(ii + 1) * 8])[0] for ii in range(len(binary) // 8)))
    grad = [[float(tmp_grad_data[ii * 3]), float(tmp_grad_data[ii * 3 + 1]), float(tmp_grad_data[ii * 3 + 2])] for ii in range(len(tmp_grad_data) // 3)]
    return np.array(grad)