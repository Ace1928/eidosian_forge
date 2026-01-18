from __future__ import annotations
import functools
from asyncio import get_running_loop
from typing import Any, Callable, Sequence, TypeVar
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.eventloop import run_in_executor_with_context
from prompt_toolkit.filters import FilterOrBool
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.key_binding.bindings.focus import focus_next, focus_previous
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.key_binding.key_bindings import KeyBindings, merge_key_bindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import AnyContainer, HSplit
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.styles import BaseStyle
from prompt_toolkit.validation import Validator
from prompt_toolkit.widgets import (
def progress_dialog(title: AnyFormattedText='', text: AnyFormattedText='', run_callback: Callable[[Callable[[int], None], Callable[[str], None]], None]=lambda *a: None, style: BaseStyle | None=None) -> Application[None]:
    """
    :param run_callback: A function that receives as input a `set_percentage`
        function and it does the work.
    """
    loop = get_running_loop()
    progressbar = ProgressBar()
    text_area = TextArea(focusable=False, height=D(preferred=10 ** 10))
    dialog = Dialog(body=HSplit([Box(Label(text=text)), Box(text_area, padding=D.exact(1)), progressbar]), title=title, with_background=True)
    app = _create_app(dialog, style)

    def set_percentage(value: int) -> None:
        progressbar.percentage = int(value)
        app.invalidate()

    def log_text(text: str) -> None:
        loop.call_soon_threadsafe(text_area.buffer.insert_text, text)
        app.invalidate()

    def start() -> None:
        try:
            run_callback(set_percentage, log_text)
        finally:
            app.exit()

    def pre_run() -> None:
        run_in_executor_with_context(start)
    app.pre_run_callables.append(pre_run)
    return app