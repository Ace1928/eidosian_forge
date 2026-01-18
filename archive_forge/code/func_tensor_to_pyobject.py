import pickle
from typing import Any
import torch
def tensor_to_pyobject(tensor: torch.Tensor) -> Any:
    nparray = tensor.cpu().numpy()
    return pickle.loads(nparray.tobytes())