from datetime import timedelta
from decimal import Decimal, ROUND_FLOOR
def fquotmod(val, low, high):
    """
    A divmod function with boundaries.

    """
    a, b = (val - low, high - low)
    div = (a / b).to_integral(ROUND_FLOOR)
    mod = a - div * b
    mod += low
    return (int(div), mod)