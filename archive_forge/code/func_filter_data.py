from __future__ import annotations
import collections.abc as c
import re
import typing as t
from .....executor import (
from .....provisioning import (
from . import (
from . import (
def filter_data(data: NamedPoints, path_filter_func: c.Callable[[str], bool], target_filter_func: c.Callable[[set[str]], set[str]]) -> NamedPoints:
    """Filter the data set using the specified filter function."""
    result: NamedPoints = {}
    for src_path, src_points in data.items():
        if not path_filter_func(src_path):
            continue
        dst_points = {}
        for src_point, src_targets in src_points.items():
            dst_targets = target_filter_func(src_targets)
            if dst_targets:
                dst_points[src_point] = dst_targets
        if dst_points:
            result[src_path] = dst_points
    return result