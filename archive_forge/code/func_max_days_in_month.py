from datetime import timedelta
from decimal import Decimal, ROUND_FLOOR
def max_days_in_month(year, month):
    """
    Determines the number of days of a specific month in a specific year.
    """
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if month in (4, 6, 9, 11):
        return 30
    if year % 400 == 0 or (year % 100 != 0 and year % 4 == 0):
        return 29
    return 28