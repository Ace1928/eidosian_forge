import os
import pathlib
import torch
from torch.jit._serialization import validate_map_location
def run_method(self, method_name, *input):
    return self._c.run_method(method_name, input)