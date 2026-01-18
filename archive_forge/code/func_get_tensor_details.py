import ctypes
import enum
import os
import platform
import sys
import numpy as np
def get_tensor_details(self):
    """Gets tensor details for every tensor with valid tensor details.

    Tensors where required information about the tensor is not found are not
    added to the list. This includes temporary tensors without a name.

    Returns:
      A list of dictionaries containing tensor information.
    """
    tensor_details = []
    for idx in range(self._interpreter.NumTensors(0)):
        try:
            tensor_details.append(self._get_tensor_details(idx, subgraph_index=0))
        except ValueError:
            pass
    return tensor_details