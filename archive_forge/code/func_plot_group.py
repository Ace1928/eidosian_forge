from __future__ import annotations
from typing import (
import numpy as np
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
from pandas.plotting._matplotlib.groupby import (
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.tools import (
def plot_group(group, ax) -> None:
    ax.hist(group.dropna().values, bins=bins, **kwargs)
    if legend:
        ax.legend()