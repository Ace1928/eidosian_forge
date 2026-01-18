import os
def mode_diff(filename, mode, **kw):
    """
    Returns the differences calculated using ``calc_mode_diff``
    """
    cur_mode = os.stat(filename).st_mode
    return calc_mode_diff(cur_mode, mode, **kw)