from __future__ import annotations
import collections.abc as c
import os
import typing as t
from .....io import (
from .....util import (
from .. import (
def expand_indexes(source_data: IndexedPoints, source_index: list[str], format_func: c.Callable[[TargetKey], TFlexKey]) -> dict[str, dict[TFlexKey, set[str]]]:
    """Expand indexes from the source into target names for easier processing of the data (arcs or lines)."""
    combined_data: dict[str, dict[TFlexKey, set[str]]] = {}
    for covered_path, covered_points in source_data.items():
        combined_points = combined_data.setdefault(covered_path, {})
        for covered_point, covered_target_indexes in covered_points.items():
            combined_point = combined_points.setdefault(format_func(covered_point), set())
            for covered_target_index in covered_target_indexes:
                combined_point.add(source_index[covered_target_index])
    return combined_data