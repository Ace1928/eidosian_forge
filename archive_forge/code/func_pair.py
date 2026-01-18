from __future__ import annotations
import io
import os
import re
import inspect
import itertools
import textwrap
from contextlib import contextmanager
from collections import abc
from collections.abc import Callable, Generator
from typing import Any, List, Literal, Optional, cast
from xml.etree import ElementTree
from cycler import cycler
import pandas as pd
from pandas import DataFrame, Series, Index
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from seaborn._marks.base import Mark
from seaborn._stats.base import Stat
from seaborn._core.data import PlotData
from seaborn._core.moves import Move
from seaborn._core.scales import Scale
from seaborn._core.subplots import Subplots
from seaborn._core.groupby import GroupBy
from seaborn._core.properties import PROPERTIES, Property
from seaborn._core.typing import (
from seaborn._core.exceptions import PlotSpecError
from seaborn._core.rules import categorical_order
from seaborn._compat import get_layout_engine, set_layout_engine
from seaborn.utils import _version_predates
from seaborn.rcmod import axes_style, plotting_context
from seaborn.palettes import color_palette
from typing import TYPE_CHECKING, TypedDict
def pair(self, x: VariableSpecList=None, y: VariableSpecList=None, wrap: int | None=None, cross: bool=True) -> Plot:
    """
        Produce subplots by pairing multiple `x` and/or `y` variables.

        Parameters
        ----------
        x, y : sequence(s) of data vectors or identifiers
            Variables that will define the grid of subplots.
        wrap : int
            When using only `x` or `y`, "wrap" subplots across a two-dimensional grid
            with this many columns (when using `x`) or rows (when using `y`).
        cross : bool
            When False, zip the `x` and `y` lists such that the first subplot gets the
            first pair, the second gets the second pair, etc. Otherwise, create a
            two-dimensional grid from the cartesian product of the lists.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.pair.rst

        """
    pair_spec: PairSpec = {}
    axes = {'x': [] if x is None else x, 'y': [] if y is None else y}
    for axis, arg in axes.items():
        if isinstance(arg, (str, int)):
            err = f'You must pass a sequence of variable keys to `{axis}`'
            raise TypeError(err)
    pair_spec['variables'] = {}
    pair_spec['structure'] = {}
    for axis in 'xy':
        keys = []
        for i, col in enumerate(axes[axis]):
            key = f'{axis}{i}'
            keys.append(key)
            pair_spec['variables'][key] = col
        if keys:
            pair_spec['structure'][axis] = keys
    if not cross and len(axes['x']) != len(axes['y']):
        err = 'Lengths of the `x` and `y` lists must match with cross=False'
        raise ValueError(err)
    pair_spec['cross'] = cross
    pair_spec['wrap'] = wrap
    new = self._clone()
    new._pair_spec.update(pair_spec)
    return new