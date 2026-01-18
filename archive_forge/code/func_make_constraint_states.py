from abc import ABC, abstractmethod
from collections import UserDict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from .beam_constraints import Constraint, ConstraintListState
def make_constraint_states(self, n):
    return [ConstraintListState([constraint.copy() for constraint in self.constraints]) for _ in range(n)]