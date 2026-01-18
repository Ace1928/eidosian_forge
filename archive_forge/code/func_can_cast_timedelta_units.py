import numpy as np
def can_cast_timedelta_units(src, dest):
    src = DATETIME_UNITS[src]
    dest = DATETIME_UNITS[dest]
    if src == dest:
        return True
    if src == 14:
        return True
    if src > dest:
        return False
    if dest == 14:
        return False
    if src <= 1 and dest > 1:
        return False
    return True