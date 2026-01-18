import warnings
from .api import _, is_validator, FancyValidator, Invalid, NoDefault
from . import declarative
from .exc import FERuntimeWarning
def merge_values(v1, v2):
    if isinstance(v1, str) and isinstance(v2, str):
        return v1 + '\n' + v2
    if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
        return merge_lists(v1, v2)
    if isinstance(v1, dict) and isinstance(v2, dict):
        return merge_dicts(v1, v2)
    return v1