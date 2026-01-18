import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def build_settables(self) -> None:
    """Create the dictionary of user-settable parameters"""

    def get_allow_style_choices(cli_self: Cmd) -> List[str]:
        """Used to tab complete allow_style values"""
        return [val.name.lower() for val in ansi.AllowStyle]

    def allow_style_type(value: str) -> ansi.AllowStyle:
        """Converts a string value into an ansi.AllowStyle"""
        try:
            return ansi.AllowStyle[value.upper()]
        except KeyError:
            raise ValueError(f'must be {ansi.AllowStyle.ALWAYS}, {ansi.AllowStyle.NEVER}, or {ansi.AllowStyle.TERMINAL} (case-insensitive)')
    self.add_settable(Settable('allow_style', allow_style_type, f'Allow ANSI text style sequences in output (valid values: {ansi.AllowStyle.ALWAYS}, {ansi.AllowStyle.NEVER}, {ansi.AllowStyle.TERMINAL})', self, choices_provider=cast(ChoicesProviderFunc, get_allow_style_choices)))
    self.add_settable(Settable('always_show_hint', bool, 'Display tab completion hint even when completion suggestions print', self))
    self.add_settable(Settable('debug', bool, 'Show full traceback on exception', self))
    self.add_settable(Settable('echo', bool, 'Echo command issued into output', self))
    self.add_settable(Settable('editor', str, "Program used by 'edit'", self))
    self.add_settable(Settable('feedback_to_output', bool, "Include nonessentials in '|', '>' results", self))
    self.add_settable(Settable('max_completion_items', int, 'Maximum number of CompletionItems to display during tab completion', self))
    self.add_settable(Settable('quiet', bool, "Don't print nonessential feedback", self))
    self.add_settable(Settable('timing', bool, 'Report execution times', self))