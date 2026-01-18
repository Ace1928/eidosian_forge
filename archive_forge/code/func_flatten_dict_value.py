from __future__ import annotations
import collections.abc
from typing import Dict, Optional
def flatten_dict_value(d: Dict, parent_key: str='') -> str:
    """
    Flatten a dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}.{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict_value(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)