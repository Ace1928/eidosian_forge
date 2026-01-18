import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
def compare_vars(map_value: Callable[[str, Any], Any]) -> List[Tuple[str, str, str]]:
    env1_set, env2_set = (set(env1_vars), set(env2_vars))
    if env1_set != env2_set:
        raise NotEqualError('field set mismatch:', [('found unique fields:', str(sorted(env1_set - env2_set)), str(sorted(env2_set - env1_set)))])
    sorted_keys = list(env1_set)
    sorted_keys.sort()
    mapped_dict = [(k, map_value(k, env1_vars[k]), map_value(k, env2_vars[k])) for k in sorted_keys]
    return [(f"{k}: values don't match.", value_to_str(val1), value_to_str(val2)) for k, val1, val2 in mapped_dict if val1 != val2]