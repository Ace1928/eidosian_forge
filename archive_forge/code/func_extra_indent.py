import textwrap
import typing as t
from contextlib import contextmanager
@contextmanager
def extra_indent(self, indent: str) -> t.Iterator[None]:
    old_initial_indent = self.initial_indent
    old_subsequent_indent = self.subsequent_indent
    self.initial_indent += indent
    self.subsequent_indent += indent
    try:
        yield
    finally:
        self.initial_indent = old_initial_indent
        self.subsequent_indent = old_subsequent_indent