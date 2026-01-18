import os
import re
import sysconfig
def is_in_range(line_range, start, end):
    """Determine if ??? is within the desired range.

    This function is used with the --range start_row end_row argument.

    Parameters
    ----------
    line_range: list
        The line number range passed to the --range argument.
    start: int
        The row number where the line under test begins in the source file.
    end: int
        The row number where the line under tests ends in the source file.

    Returns
    -------
    in_range : bool
        True if in range or range is None, else False
    """
    if line_range is None:
        return True
    return any((line_range[0] <= line_no <= line_range[1] for line_no in range(start, end + 1)))