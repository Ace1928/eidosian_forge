import collections
import copy
import functools
from importlib import resources
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
def make_bond_key(atom1_name: str, atom2_name: str) -> str:
    """Unique key to lookup bonds."""
    return '-'.join(sorted([atom1_name, atom2_name]))