from __future__ import annotations
import re
import textwrap
import traceback
import typing as t
from .util import (
def process_range(self) -> None:
    """Process a diff range line."""
    match = re.search('^@@ -((?P<old_start>[0-9]+),)?(?P<old_count>[0-9]+) \\+((?P<new_start>[0-9]+),)?(?P<new_count>[0-9]+) @@', self.line)
    if not match:
        raise Exception('Unexpected diff range line.')
    self.file.old.set_start(int(match.group('old_start') or 1), int(match.group('old_count')))
    self.file.new.set_start(int(match.group('new_start') or 1), int(match.group('new_count')))
    self.action = self.process_content