from __future__ import annotations
import typing as t
from .....executor import (
from .....provisioning import (
from . import (
from . import (
def merge_indexes(source_data: IndexedPoints, source_index: list[str], combined_data: IndexedPoints, combined_index: TargetIndexes) -> None:
    """Merge indexes from the source into the combined data set (arcs or lines)."""
    for covered_path, covered_points in source_data.items():
        combined_points = combined_data.setdefault(covered_path, {})
        for covered_point, covered_target_indexes in covered_points.items():
            combined_point = combined_points.setdefault(covered_point, set())
            for covered_target_index in covered_target_indexes:
                combined_point.add(get_target_index(source_index[covered_target_index], combined_index))