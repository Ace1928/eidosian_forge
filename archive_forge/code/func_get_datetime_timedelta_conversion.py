import numpy as np
def get_datetime_timedelta_conversion(datetime_unit, timedelta_unit):
    """
    Compute a possible conversion for combining *datetime_unit* and
    *timedelta_unit* (presumably for adding or subtracting).
    Return (result unit, integer datetime multiplier, integer timedelta
    multiplier). RuntimeError is raised if the combination is impossible.
    """
    dt_unit_code = DATETIME_UNITS[datetime_unit]
    td_unit_code = DATETIME_UNITS[timedelta_unit]
    if td_unit_code == 14 or dt_unit_code == 14:
        return (datetime_unit, 1, 1)
    if td_unit_code < 2 and dt_unit_code >= 2:
        raise RuntimeError('cannot combine datetime64(%r) and timedelta64(%r)' % (datetime_unit, timedelta_unit))
    dt_factor, td_factor = (1, 1)
    if dt_unit_code == 0:
        if td_unit_code >= 4:
            dt_factor = 97 + 400 * 365
            td_factor = 400
            dt_unit_code = 4
        elif td_unit_code == 2:
            dt_factor = 97 + 400 * 365
            td_factor = 400 * 7
            dt_unit_code = 2
    elif dt_unit_code == 1:
        if td_unit_code >= 4:
            dt_factor = 97 + 400 * 365
            td_factor = 400 * 12
            dt_unit_code = 4
        elif td_unit_code == 2:
            dt_factor = 97 + 400 * 365
            td_factor = 400 * 12 * 7
            dt_unit_code = 2
    if td_unit_code >= dt_unit_code:
        factor = _get_conversion_multiplier(dt_unit_code, td_unit_code)
        assert factor is not None, (dt_unit_code, td_unit_code)
        return (timedelta_unit, dt_factor * factor, td_factor)
    else:
        factor = _get_conversion_multiplier(td_unit_code, dt_unit_code)
        assert factor is not None, (dt_unit_code, td_unit_code)
        return (datetime_unit, dt_factor, td_factor * factor)