from __future__ import annotations
import collections.abc as c
import os
import typing as t
from .....io import (
from .....util import (
from .. import (
def generate_indexes(target_indexes: TargetIndexes, data: NamedPoints) -> IndexedPoints:
    """Return an indexed version of the given data (arcs or points)."""
    results: IndexedPoints = {}
    for path, points in data.items():
        result_points = results[path] = {}
        for point, target_names in points.items():
            result_point = result_points[point] = set()
            for target_name in target_names:
                result_point.add(get_target_index(target_name, target_indexes))
    return results