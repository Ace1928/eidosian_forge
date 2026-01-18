from sys import maxunicode
from typing import Iterable, Iterator, Optional, Set, Tuple, Union
def code_point_reverse_order(cp: CodePoint) -> int:
    """Reverse ordering function for code points."""
    return cp if isinstance(cp, int) else cp[1] - 1