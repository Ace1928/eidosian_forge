from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import AsyncGenerator, Callable, Iterable, Sequence
from prompt_toolkit.document import Document
from prompt_toolkit.eventloop import aclosing, generator_to_async_generator
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText, StyleAndTextTuples
@property
def display_text(self) -> str:
    """The 'display' field as plain text."""
    from prompt_toolkit.formatted_text import fragment_list_to_text
    return fragment_list_to_text(self.display)