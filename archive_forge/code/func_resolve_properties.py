from __future__ import annotations
from dataclasses import dataclass, fields, field
import textwrap
from typing import Any, Callable, Union
from collections.abc import Generator
import numpy as np
import pandas as pd
import matplotlib as mpl
from numpy import ndarray
from pandas import DataFrame
from matplotlib.artist import Artist
from seaborn._core.scales import Scale
from seaborn._core.properties import (
from seaborn._core.exceptions import PlotSpecError
def resolve_properties(mark: Mark, data: DataFrame, scales: dict[str, Scale]) -> dict[str, Any]:
    props = {name: mark._resolve(data, name, scales) for name in mark._mappable_props}
    return props