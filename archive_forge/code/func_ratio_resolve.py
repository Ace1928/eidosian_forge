import sys
from fractions import Fraction
from math import ceil
from typing import cast, List, Optional, Sequence
def ratio_resolve(total: int, edges: Sequence[Edge]) -> List[int]:
    """Divide total space to satisfy size, ratio, and minimum_size, constraints.

    The returned list of integers should add up to total in most cases, unless it is
    impossible to satisfy all the constraints. For instance, if there are two edges
    with a minimum size of 20 each and `total` is 30 then the returned list will be
    greater than total. In practice, this would mean that a Layout object would
    clip the rows that would overflow the screen height.

    Args:
        total (int): Total number of characters.
        edges (List[Edge]): Edges within total space.

    Returns:
        List[int]: Number of characters for each edge.
    """
    sizes = [edge.size or None for edge in edges]
    _Fraction = Fraction
    while None in sizes:
        flexible_edges = [(index, edge) for index, (size, edge) in enumerate(zip(sizes, edges)) if size is None]
        remaining = total - sum((size or 0 for size in sizes))
        if remaining <= 0:
            return [edge.minimum_size or 1 if size is None else size for size, edge in zip(sizes, edges)]
        portion = _Fraction(remaining, sum((edge.ratio or 1 for _, edge in flexible_edges)))
        for index, edge in flexible_edges:
            if portion * edge.ratio <= edge.minimum_size:
                sizes[index] = edge.minimum_size
                break
        else:
            remainder = _Fraction(0)
            for index, edge in flexible_edges:
                size, remainder = divmod(portion * edge.ratio + remainder, 1)
                sizes[index] = size
            break
    return cast(List[int], sizes)