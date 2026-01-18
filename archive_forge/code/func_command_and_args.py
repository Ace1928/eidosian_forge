import re
import shlex
from typing import (
import attr
from . import (
from .exceptions import (
@property
def command_and_args(self) -> str:
    """Combine command and args with a space separating them.

        Quoted arguments remain quoted. Output redirection and piping are
        excluded, as are any command terminators.
        """
    if self.command and self.args:
        rtn = f'{self.command} {self.args}'
    elif self.command:
        rtn = self.command
    else:
        rtn = ''
    return rtn