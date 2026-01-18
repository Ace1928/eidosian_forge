import itertools
import math
from typing import Sequence, Tuple, Union
import numpy as np
from onnx.reference.op_run import OpRun
def lp_pool(x: np.array, p: int) -> float:
    y = 0
    for v in np.nditer(x):
        y += abs(v) ** p
    return y ** (1.0 / p)