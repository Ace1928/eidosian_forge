from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union, TYPE_CHECKING
import re
import numpy as np
from cirq import ops, linalg, protocols, value
def output_line_gap(n):
    line_gap[0] = max(line_gap[0], n)