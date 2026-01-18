from __future__ import annotations
from asyncio.events import AbstractEventLoop
from typing import TYPE_CHECKING, Any, TextIO
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app_or_none, get_app_session
from prompt_toolkit.application.run_in_terminal import run_in_terminal
from prompt_toolkit.formatted_text import (
from prompt_toolkit.input import DummyInput
from prompt_toolkit.layout import Layout
from prompt_toolkit.output import ColorDepth, Output
from prompt_toolkit.output.defaults import create_output
from prompt_toolkit.renderer import (
from prompt_toolkit.styles import (
def print_container(container: AnyContainer, file: TextIO | None=None, style: BaseStyle | None=None, include_default_pygments_style: bool=True) -> None:
    """
    Print any layout to the output in a non-interactive way.

    Example usage::

        from prompt_toolkit.widgets import Frame, TextArea
        print_container(
            Frame(TextArea(text='Hello world!')))
    """
    if file:
        output = create_output(stdout=file)
    else:
        output = get_app_session().output
    app: Application[None] = Application(layout=Layout(container=container), output=output, input=DummyInput(), style=_create_merged_style(style, include_default_pygments_style=include_default_pygments_style))
    try:
        app.run(in_thread=True)
    except EOFError:
        pass