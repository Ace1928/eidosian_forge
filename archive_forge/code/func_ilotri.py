from __future__ import annotations
import itertools
from typing import TYPE_CHECKING
def ilotri(items: Iterable[Iterable], diago: bool=True, with_inds: bool=False) -> Iterable[Iterable]:
    """
    A generator that yields the lower triangle of the matrix (items x items)

    Args:
        items: Iterable object with elements [e0, e1, ...]
        diago: False if diagonal matrix elements should be excluded
        with_inds: If True, (i,j) (e_i, e_j) is returned else (e_i, e_j)

    >>> for (ij, mate) in ilotri([0,1], with_inds=True):
    ...     print("ij:", ij, "mate:", mate)
    ij: (0, 0) mate: (0, 0)
    ij: (1, 0) mate: (1, 0)
    ij: (1, 1) mate: (1, 1)
    """
    for ii, item1 in enumerate(items):
        for jj, item2 in enumerate(items):
            do_yield = jj <= ii if diago else jj < ii
            if do_yield:
                if with_inds:
                    yield ((ii, jj), (item1, item2))
                else:
                    yield (item1, item2)