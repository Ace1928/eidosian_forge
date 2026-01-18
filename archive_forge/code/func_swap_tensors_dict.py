from typing import Dict, Iterable, List, Tuple
import torch
def swap_tensors_dict(self, named_tensors: Dict[str, torch.Tensor], allow_missing: bool=False) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
        Swap the attributes specified by the given paths to values.

        For example, to swap the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.swap_tensors_dict({
            "layer1.conv1.weight": weight,
            "layer1.conv1.bias": bias,
        })
        """
    orig_named_tensors = {}
    missing_keys = []
    try:
        for name, tensor in named_tensors.items():
            orig_tensor = self.swap_tensor(name, tensor, allow_missing=True)
            if orig_tensor is _MISSING:
                missing_keys.append(name)
            orig_named_tensors[name] = orig_tensor
    except Exception:
        for name, orig_tensor in orig_named_tensors.items():
            self.swap_tensor(name, orig_tensor, allow_missing=True)
        raise
    if missing_keys and (not allow_missing):
        for name, orig_tensor in orig_named_tensors.items():
            self.swap_tensor(name, orig_tensor, allow_missing=True)
        raise RuntimeError(f'Missing key(s): {', '.join(map(repr, missing_keys))}.')
    return (orig_named_tensors, missing_keys)