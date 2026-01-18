from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Callable, Optional, Union, cast
import numpy as np
from pandas import DataFrame
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._core.typing import Default
def groupby_pos(s):
    grouper = [groups[v] for v in [orient, 'col', 'row'] if v in data]
    return s.groupby(grouper, sort=False, observed=True)