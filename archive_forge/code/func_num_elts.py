from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import numpy.typing as npt
def num_elts(self) -> int:
    return prod(self.dimensions)