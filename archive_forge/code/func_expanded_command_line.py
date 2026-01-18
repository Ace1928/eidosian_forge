import re
import shlex
from typing import (
import attr
from . import (
from .exceptions import (
@property
def expanded_command_line(self) -> str:
    """Concatenate :meth:`~cmd2.Statement.command_and_args`
        and :meth:`~cmd2.Statement.post_command`"""
    return self.command_and_args + self.post_command