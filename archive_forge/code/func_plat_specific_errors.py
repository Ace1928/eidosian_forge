import errno
import sys
def plat_specific_errors(*errnames):
    """Return error numbers for all errors in ``errnames`` on this platform.

    The :py:mod:`errno` module contains different global constants
    depending on the specific platform (OS). This function will return
    the list of numeric values for a given list of potential names.
    """
    missing_attr = {None}
    unique_nums = {getattr(errno, k, None) for k in errnames}
    return list(unique_nums - missing_attr)