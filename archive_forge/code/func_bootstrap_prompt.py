from __future__ import with_statement
import click
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from ._completer import ClickCompleter
from .exceptions import ClickExit  # type: ignore[attr-defined]
from .exceptions import CommandLineParserError, ExitReplException, InvalidGroupFormat
from .utils import _execute_internal_and_sys_cmds
def bootstrap_prompt(group, prompt_kwargs, ctx=None):
    """
    Bootstrap prompt_toolkit kwargs or use user defined values.

    :param group: click Group
    :param prompt_kwargs: The user specified prompt kwargs.
    """
    defaults = {'history': InMemoryHistory(), 'completer': ClickCompleter(group, ctx=ctx), 'message': '> '}
    defaults.update(prompt_kwargs)
    return defaults