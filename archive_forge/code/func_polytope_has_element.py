from __future__ import annotations
from copy import copy
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
from qiskit.exceptions import QiskitError
from .utilities import EPSILON
def polytope_has_element(polytope, point):
    """
    Tests whether `polytope` contains `point.
    """
    return all((-EPSILON <= inequality[0] + sum((x * y for x, y in zip(point, inequality[1:]))) for inequality in polytope.inequalities)) and all((abs(equality[0] + sum((x * y for x, y in zip(point, equality[1:])))) <= EPSILON for equality in polytope.equalities))