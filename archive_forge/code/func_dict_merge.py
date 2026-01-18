import logging
import sys
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, Mapping, Union
def dict_merge(*dicts: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Merge all provided dicts into 1 dict.
    *dicts : `dict`
        dictionaries to merge
    """
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged