from __future__ import annotations
import re
import textwrap
import traceback
import typing as t
from .util import (
def process_content(self) -> None:
    """Process a diff content line."""
    if self.line == '\\ No newline at end of file':
        if self.previous_line.startswith(' '):
            self.file.old.eof_newline = False
            self.file.new.eof_newline = False
        elif self.previous_line.startswith('-'):
            self.file.old.eof_newline = False
        elif self.previous_line.startswith('+'):
            self.file.new.eof_newline = False
        else:
            raise Exception('Unexpected previous diff content line.')
        return
    if self.file.is_complete:
        self.process_continue()
        return
    if self.line.startswith(' '):
        self.file.old.append(self.line)
        self.file.new.append(self.line)
    elif self.line.startswith('-'):
        self.file.old.append(self.line)
    elif self.line.startswith('+'):
        self.file.new.append(self.line)
    else:
        raise Exception('Unexpected diff content line.')