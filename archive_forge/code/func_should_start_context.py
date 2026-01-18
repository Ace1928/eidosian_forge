from __future__ import annotations
from types import FrameType
from typing import cast, Callable, Sequence
def should_start_context(frame: FrameType) -> str | None:
    """The combiner for multiple context switchers."""
    for switcher in context_switchers:
        new_context = switcher(frame)
        if new_context is not None:
            return new_context
    return None