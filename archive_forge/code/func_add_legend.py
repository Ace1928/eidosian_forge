from __future__ import annotations
import functools
import itertools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, cast
import numpy as np
from xarray.core.formatting import format_item
from xarray.core.types import HueStyleOptions, T_DataArrayOrSet
from xarray.plot.utils import (
def add_legend(self, *, label: str | None=None, use_legend_elements: bool=False, **kwargs: Any) -> None:
    if use_legend_elements:
        self.figlegend = _add_legend(**kwargs)
    else:
        self.figlegend = self.fig.legend(handles=self._mappables[-1], labels=list(self._hue_var.to_numpy()), title=label if label is not None else label_from_attrs(self._hue_var), loc=kwargs.pop('loc', 'center right'), **kwargs)
    self._adjust_fig_for_guide(self.figlegend)