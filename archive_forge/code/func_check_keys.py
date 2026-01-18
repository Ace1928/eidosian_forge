from typing import Dict, Iterable, List, Tuple
import torch
def check_keys(self, keys: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Check that the given keys are valid."""
    keys = set(keys)
    valid_keys = {name for name, _ in self.named_tensors(remove_duplicate=False)}
    missing_keys = valid_keys - keys
    unexpected_keys = keys - valid_keys
    return (sorted(missing_keys), sorted(unexpected_keys))