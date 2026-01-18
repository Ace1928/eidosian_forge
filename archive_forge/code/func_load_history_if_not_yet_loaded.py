from __future__ import annotations
import asyncio
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections import deque
from enum import Enum
from functools import wraps
from typing import Any, Callable, Coroutine, Iterable, TypeVar, cast
from .application.current import get_app
from .application.run_in_terminal import run_in_terminal
from .auto_suggest import AutoSuggest, Suggestion
from .cache import FastDictCache
from .clipboard import ClipboardData
from .completion import (
from .document import Document
from .eventloop import aclosing
from .filters import FilterOrBool, to_filter
from .history import History, InMemoryHistory
from .search import SearchDirection, SearchState
from .selection import PasteMode, SelectionState, SelectionType
from .utils import Event, to_str
from .validation import ValidationError, Validator
def load_history_if_not_yet_loaded(self) -> None:
    """
        Create task for populating the buffer history (if not yet done).

        Note::

            This needs to be called from within the event loop of the
            application, because history loading is async, and we need to be
            sure the right event loop is active. Therefor, we call this method
            in the `BufferControl.create_content`.

            There are situations where prompt_toolkit applications are created
            in one thread, but will later run in a different thread (Ptpython
            is one example. The REPL runs in a separate thread, in order to
            prevent interfering with a potential different event loop in the
            main thread. The REPL UI however is still created in the main
            thread.) We could decide to not support creating prompt_toolkit
            objects in one thread and running the application in a different
            thread, but history loading is the only place where it matters, and
            this solves it.
        """
    if self._load_history_task is None:

        async def load_history() -> None:
            async for item in self.history.load():
                self._working_lines.appendleft(item)
                self.__working_index += 1
        self._load_history_task = get_app().create_background_task(load_history())

        def load_history_done(f: asyncio.Future[None]) -> None:
            """
                Handle `load_history` result when either done, cancelled, or
                when an exception was raised.
                """
            try:
                f.result()
            except asyncio.CancelledError:
                pass
            except GeneratorExit:
                pass
            except BaseException:
                logger.exception('Loading history failed')
        self._load_history_task.add_done_callback(load_history_done)