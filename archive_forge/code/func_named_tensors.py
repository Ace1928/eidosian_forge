from typing import Dict, Iterable, List, Tuple
import torch
def named_tensors(self, remove_duplicate: bool=True) -> Iterable[Tuple[str, torch.Tensor]]:
    """Iterate over all the tensors in the module."""
    yield from self.module.named_parameters(remove_duplicate=remove_duplicate)
    yield from self.module.named_buffers(remove_duplicate=remove_duplicate)