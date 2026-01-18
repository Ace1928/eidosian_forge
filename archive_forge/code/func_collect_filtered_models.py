from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Callable
from ..core.has_props import HasProps, Qualified
from ..util.dataclasses import entries, is_dataclass
def collect_filtered_models(discard: Callable[[Model], bool] | None, *input_values: Any) -> list[Model]:
    """ Collect a duplicate-free list of all other Bokeh models referred to by
    this model, or by any of its references, etc, unless filtered-out by the
    provided callable.

    Iterate over ``input_values`` and descend through their structure
    collecting all nested ``Models`` on the go.

    Args:
        *discard (Callable[[Model], bool])
            a callable which accepts a *Model* instance as its single argument
            and returns a boolean stating whether to discard the instance. The
            latter means that the instance will not be added to collected
            models nor will its references be explored.

        *input_values (Model)
            Bokeh models to collect other models from

    Returns:
        list(Model)

    """
    ids: set[ID] = set()
    collected: list[Model] = []
    queued: list[Model] = []

    def queue_one(obj: Model) -> None:
        if obj.id not in ids and (not (callable(discard) and discard(obj))):
            queued.append(obj)
    for value in input_values:
        visit_value_and_its_immediate_references(value, queue_one)
    while queued:
        obj = queued.pop(0)
        if obj.id not in ids:
            ids.add(obj.id)
            collected.append(obj)
            visit_immediate_value_references(obj, queue_one)
    return collected