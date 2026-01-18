import itertools
from collections import OrderedDict
import numpy as np
def is_valid_einsum_char(x):
    """Check if the character ``x`` is valid for numpy einsum.

    Examples
    --------
    >>> is_valid_einsum_char("a")
    True

    >>> is_valid_einsum_char("Ǵ")
    False
    """
    return x in _einsum_symbols_base or x in ',->.'