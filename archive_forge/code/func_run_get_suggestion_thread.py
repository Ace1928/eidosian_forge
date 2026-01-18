from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable
from prompt_toolkit.eventloop import run_in_executor_with_context
from .document import Document
from .filters import Filter, to_filter
def run_get_suggestion_thread() -> Suggestion | None:
    return self.get_suggestion(buff, document)