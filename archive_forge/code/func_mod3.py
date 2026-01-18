from decimal import ROUND_FLOOR, Decimal
def mod3(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    dividend = value - low
    divisor = high - low
    return mod2(dividend, divisor) + low