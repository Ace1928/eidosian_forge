from functools import reduce
from operator import mul
from typing import List, Tuple
def sum_shapes(shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Give the shape resulting from summing a list of shapes.

    Summation semantics are exactly the same as NumPy's, including
    broadcasting.

    Parameters
    ----------
    shapes : list of tuple
        The shapes to sum.

    Returns
    -------
    tuple
        The shape of the sum.

    Raises
    ------
    ValueError
        If the shapes are not compatible.
    """
    shape = shapes[0]
    for t in shapes[1:]:
        if shape != t and len(squeezed(shape)) != 0 and (len(squeezed(t)) != 0):
            raise ValueError('Cannot broadcast dimensions ' + len(shapes) * ' %s' % tuple(shapes))
        longer = shape if len(shape) >= len(t) else t
        shorter = shape if len(shape) < len(t) else t
        offset = len(longer) - len(shorter)
        prefix = list(longer[:offset])
        suffix = []
        for d1, d2 in zip(reversed(longer[offset:]), reversed(shorter)):
            if d1 != d2 and (not (d1 == 1 or d2 == 1)):
                raise ValueError('Incompatible dimensions' + len(shapes) * ' %s' % tuple(shapes))
            new_dim = d1 if d1 >= d2 else d2
            suffix = [new_dim] + suffix
        shape = tuple(prefix + suffix)
    return tuple(shape)