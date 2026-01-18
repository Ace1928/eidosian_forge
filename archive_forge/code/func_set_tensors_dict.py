from typing import Dict, Iterable, List, Tuple
import torch
def set_tensors_dict(self, named_tensors: Dict[str, torch.Tensor]) -> None:
    """
        Set the attributes specified by the given paths to values.

        For example, to set the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.set_tensors_dict({
            "layer1.conv1.weight": weight,
            "layer1.conv1.bias": bias,
        })
        """
    for name, value in named_tensors.items():
        self.set_tensor(name, value)