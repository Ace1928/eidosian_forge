from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Tuple, Union, cast
from prompt_toolkit.mouse_events import MouseEvent
def merge_formatted_text(items: Iterable[AnyFormattedText]) -> AnyFormattedText:
    """
    Merge (Concatenate) several pieces of formatted text together.
    """

    def _merge_formatted_text() -> AnyFormattedText:
        result = FormattedText()
        for i in items:
            result.extend(to_formatted_text(i))
        return result
    return _merge_formatted_text