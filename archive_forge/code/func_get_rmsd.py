from __future__ import annotations
import abc
import copy
import itertools
import logging
import math
import re
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pymatgen.core.structure import Molecule
def get_rmsd(self, mol1, mol2):
    """
        Get RMSD between two molecule with arbitrary atom order.

        Returns:
            RMSD if topology of the two molecules are the same
            Infinite if  the topology is different
        """
    label1, label2 = self._mapper.uniform_labels(mol1, mol2)
    if label1 is None or label2 is None:
        return float('Inf')
    return self._calc_rms(mol1, mol2, label1, label2)