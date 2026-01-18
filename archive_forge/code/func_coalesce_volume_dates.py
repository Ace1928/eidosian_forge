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
def coalesce_volume_dates(in_volumes, in_dates, indexes):
    """Sums volumes between the indexes and ouputs
    dates at the indexes
    in_volumes : original volume list
    in_dates : original dates list
    indexes : list of indexes

    Returns
    -------
    volumes: new volume array
    dates: new dates array
    """
    volumes, dates = ([], [])
    for i in range(len(indexes)):
        dates.append(in_dates[indexes[i]])
        to_sum_to = indexes[i + 1] if i + 1 < len(indexes) else len(in_volumes)
        volumes.append(sum(in_volumes[indexes[i]:to_sum_to]))
    return (volumes, dates)