from __future__ import annotations
from types import FrameType
from typing import cast, Callable, Sequence
def combine_context_switchers(context_switchers: Sequence[Callable[[FrameType], str | None]]) -> Callable[[FrameType], str | None] | None:
    """Create a single context switcher from multiple switchers.

    `context_switchers` is a list of functions that take a frame as an
    argument and return a string to use as the new context label.

    Returns a function that composites `context_switchers` functions, or None
    if `context_switchers` is an empty list.

    When invoked, the combined switcher calls `context_switchers` one-by-one
    until a string is returned.  The combined switcher returns None if all
    `context_switchers` return None.
    """
    if not context_switchers:
        return None
    if len(context_switchers) == 1:
        return context_switchers[0]

    def should_start_context(frame: FrameType) -> str | None:
        """The combiner for multiple context switchers."""
        for switcher in context_switchers:
            new_context = switcher(frame)
            if new_context is not None:
                return new_context
        return None
    return should_start_context