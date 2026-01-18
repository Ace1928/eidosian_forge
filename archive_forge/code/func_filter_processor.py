from a buffer before the BufferControl will render it to the screen.
from __future__ import annotations
import re
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable, Hashable, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.document import Document
from prompt_toolkit.filters import FilterOrBool, to_filter, vi_insert_multiple_mode
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import fragment_list_len, fragment_list_to_text
from prompt_toolkit.search import SearchDirection
from prompt_toolkit.utils import to_int, to_str
from .utils import explode_text_fragments
def filter_processor(item: Processor) -> Processor | None:
    """Filter processors from the main control that we want to disable
            here. This returns either an accepted processor or None."""
    if isinstance(item, _MergedProcessor):
        accepted_processors = [filter_processor(p) for p in item.processors]
        return merge_processors([p for p in accepted_processors if p is not None])
    elif isinstance(item, ConditionalProcessor):
        p = filter_processor(item.processor)
        if p:
            return ConditionalProcessor(p, item.filter)
    elif not isinstance(item, excluded_processors):
        return item
    return None