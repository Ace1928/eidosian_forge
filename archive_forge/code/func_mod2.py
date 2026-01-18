from decimal import ROUND_FLOOR, Decimal
def mod2(dividend: Decimal, divisor: Decimal) -> Decimal:
    return dividend - quot2(dividend, divisor) * divisor