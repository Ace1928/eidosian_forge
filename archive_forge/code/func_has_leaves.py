from enum import IntEnum
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
def has_leaves(self):
    return self._leafs_ is not None and len(self._leafs_) > 0