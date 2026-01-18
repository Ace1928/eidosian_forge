from typing import Dict, Iterable, List, Tuple
import torch
def swap_tensor(self, name: str, value: torch.Tensor, allow_missing: bool=False) -> torch.Tensor:
    """
        Swap the attribute specified by the given path to value.

        For example, to swap the attribute mod.layer1.conv1.weight,
        use accessor.swap_tensor("layer1.conv1.weight", value)
        """
    prefix, _, attr = name.rpartition('.')
    return swap_tensor(self.get_submodule(prefix), attr, value, allow_missing=allow_missing)