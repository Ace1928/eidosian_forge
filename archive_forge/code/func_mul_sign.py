from typing import TYPE_CHECKING, List, Tuple
def mul_sign(lh_expr: 'Expression', rh_expr: 'Expression') -> Tuple[bool, bool]:
    """Give the sign resulting from multiplying two expressions.

    Args:
        lh_expr: An expression.
        rh_expr: An expression.

    Returns:
        The sign (is pos, is neg) of the product.
    """
    lh_nonneg = lh_expr.is_nonneg()
    rh_nonneg = rh_expr.is_nonneg()
    lh_nonpos = lh_expr.is_nonpos()
    rh_nonpos = rh_expr.is_nonpos()
    lh_zero = lh_nonneg and lh_nonpos
    rh_zero = rh_nonneg and rh_nonpos
    is_zero = lh_zero or rh_zero
    is_pos = is_zero or (lh_nonneg and rh_nonneg) or (lh_nonpos and rh_nonpos)
    is_neg = is_zero or (lh_nonneg and rh_nonpos) or (lh_nonpos and rh_nonneg)
    return (is_pos, is_neg)