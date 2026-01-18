import inspect
from typing import Dict, List, Union
from torch import _C
from torch.onnx import _constants
from torch.onnx._internal import registration
def is_backward(self) -> bool:
    return 'backward' in self.name