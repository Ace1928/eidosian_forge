import numpy as np
def get_timedelta_conversion_factor(src_unit, dest_unit):
    """
    Return an integer multiplier allowing to convert from timedeltas
    of *src_unit* to *dest_unit*.
    """
    return _get_conversion_multiplier(DATETIME_UNITS[src_unit], DATETIME_UNITS[dest_unit])