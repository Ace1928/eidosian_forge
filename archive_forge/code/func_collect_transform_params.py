import json
import anywidget
import traitlets
import pathlib
from typing import Any, Set, Optional
import altair as alt
from altair.utils._vegafusion_data import (
from altair import TopLevelSpec
from altair.utils.selection import IndexSelection, PointSelection, IntervalSelection
def collect_transform_params(chart: TopLevelSpec) -> Set[str]:
    """
    Collect the names of params that are defined by transforms

    Parameters
    ----------
    chart: Chart from which to extract transform params

    Returns
    -------
    set of param names
    """
    transform_params = set()
    for prop in ('layer', 'concat', 'hconcat', 'vconcat'):
        for child in getattr(chart, prop, []):
            transform_params.update(collect_transform_params(child))
    transforms = getattr(chart, 'transform', [])
    transforms = transforms if transforms != alt.Undefined else []
    for tx in transforms:
        if hasattr(tx, 'param'):
            transform_params.add(tx.param)
    return transform_params