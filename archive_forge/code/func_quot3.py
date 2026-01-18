from decimal import ROUND_FLOOR, Decimal
def quot3(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    dividend = value - low
    divisor = high - low
    return (dividend / divisor).to_integral_value(rounding=ROUND_FLOOR)