import functools
from dataclasses import asdict, dataclass, field, replace
from itertools import zip_longest
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from outlines.base import vectorize
from outlines.caching import cache
def build_optimistic_mask(transposed: List[Set[int]], max_mask_size: int=300) -> Dict[int, int]:
    """We build the largest mask possible.

    Tokens are added from left to right, so if the encoded choices are e.g.
    `[[1,2], [3,4]]`, `1` and `3` will be added before `2` and `4`.

    Parameters
    ----------
    transposed
        A list of lists that contain the nth token of each choice.

    """
    mask: Dict[int, int] = {}
    for tokens in transposed:
        for token in tokens:
            if len(mask) == max_mask_size:
                return mask
            mask[token] = 100
    return mask