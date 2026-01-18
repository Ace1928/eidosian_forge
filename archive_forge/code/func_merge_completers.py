from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import AsyncGenerator, Callable, Iterable, Sequence
from prompt_toolkit.document import Document
from prompt_toolkit.eventloop import aclosing, generator_to_async_generator
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText, StyleAndTextTuples
def merge_completers(completers: Sequence[Completer], deduplicate: bool=False) -> Completer:
    """
    Combine several completers into one.

    :param deduplicate: If `True`, wrap the result in a `DeduplicateCompleter`
        so that completions that would result in the same text will be
        deduplicated.
    """
    if deduplicate:
        from .deduplicate import DeduplicateCompleter
        return DeduplicateCompleter(_MergedCompleter(completers))
    return _MergedCompleter(completers)