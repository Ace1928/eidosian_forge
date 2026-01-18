from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Generator
@contextmanager
def create_app_session_from_tty() -> Generator[AppSession, None, None]:
    """
    Create `AppSession` that always prefers the TTY input/output.

    Even if `sys.stdin` and `sys.stdout` are connected to input/output pipes,
    this will still use the terminal for interaction (because `sys.stderr` is
    still connected to the terminal).

    Usage::

        from prompt_toolkit.shortcuts import prompt

        with create_app_session_from_tty():
            prompt('>')
    """
    from prompt_toolkit.input.defaults import create_input
    from prompt_toolkit.output.defaults import create_output
    input = create_input(always_prefer_tty=True)
    output = create_output(always_prefer_tty=True)
    with create_app_session(input=input, output=output) as app_session:
        yield app_session