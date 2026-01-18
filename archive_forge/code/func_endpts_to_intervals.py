from collections.abc import Sequence
from plotly import exceptions
from plotly.colors import (
def endpts_to_intervals(endpts):
    """
    Returns a list of intervals for categorical colormaps

    Accepts a list or tuple of sequentially increasing numbers and returns
    a list representation of the mathematical intervals with these numbers
    as endpoints. For example, [1, 6] returns [[-inf, 1], [1, 6], [6, inf]]

    :raises: (PlotlyError) If input is not a list or tuple
    :raises: (PlotlyError) If the input contains a string
    :raises: (PlotlyError) If any number does not increase after the
        previous one in the sequence
    """
    length = len(endpts)
    if not (isinstance(endpts, tuple) or isinstance(endpts, list)):
        raise exceptions.PlotlyError('The intervals_endpts argument must be a list or tuple of a sequence of increasing numbers.')
    for item in endpts:
        if isinstance(item, str):
            raise exceptions.PlotlyError('The intervals_endpts argument must be a list or tuple of a sequence of increasing numbers.')
    for k in range(length - 1):
        if endpts[k] >= endpts[k + 1]:
            raise exceptions.PlotlyError('The intervals_endpts argument must be a list or tuple of a sequence of increasing numbers.')
    else:
        intervals = []
        intervals.append([float('-inf'), endpts[0]])
        for k in range(length - 1):
            interval = []
            interval.append(endpts[k])
            interval.append(endpts[k + 1])
            intervals.append(interval)
        intervals.append([endpts[length - 1], float('inf')])
        return intervals