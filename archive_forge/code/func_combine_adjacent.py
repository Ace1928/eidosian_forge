import numpy  as np
import pandas as pd
import matplotlib.dates as mdates
import datetime
from itertools import cycle
from matplotlib import colors as mcolors, pyplot as plt
from matplotlib.patches     import Ellipse
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from mplfinance._arg_validators import _alines_validator, _bypass_kwarg_validation
from mplfinance._arg_validators import _xlim_validator, _is_datelike
from mplfinance._styles         import _get_mpfstyle
from mplfinance._helpers        import _mpf_to_rgba
from six.moves import zip
from matplotlib.ticker import Formatter
def combine_adjacent(arr):
    """Sum like signed adjacent elements
    arr : starting array

    Returns
    -------
    output: new summed array
    indexes: indexes indicating the first
             element summed for each group in arr
    """
    output, indexes = ([], [])
    curr_i = 0
    while len(arr) > 0:
        curr_sign = arr[0] / abs(arr[0])
        index = 0
        while index < len(arr) and arr[index] / abs(arr[index]) == curr_sign:
            index += 1
        output.append(sum(arr[:index]))
        indexes.append(curr_i)
        curr_i += index
        for _ in range(index):
            arr.pop(0)
    return (output, indexes)