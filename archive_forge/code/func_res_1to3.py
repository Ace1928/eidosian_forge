import dataclasses
import re
import string
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from . import residue_constants
def res_1to3(r: int) -> str:
    return residue_constants.restype_1to3.get(restypes[r], 'UNK')