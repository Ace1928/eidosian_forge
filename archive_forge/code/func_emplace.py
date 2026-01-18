from enum import IntEnum
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
def emplace(self, key, value):
    return self.leafs_.emplace(key, value)