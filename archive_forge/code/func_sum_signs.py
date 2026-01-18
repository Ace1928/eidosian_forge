from typing import TYPE_CHECKING, List, Tuple
def sum_signs(exprs: List['Expression']) -> Tuple[bool, bool]:
    """Give the sign resulting from summing a list of expressions.

    Args:
        shapes: A list of sign (is pos, is neg) tuples.

    Returns:
        The sign (is pos, is neg) of the sum.
    """
    is_pos = all((expr.is_nonneg() for expr in exprs))
    is_neg = all((expr.is_nonpos() for expr in exprs))
    return (is_pos, is_neg)