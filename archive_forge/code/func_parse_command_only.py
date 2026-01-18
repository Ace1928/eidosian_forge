import re
import shlex
from typing import (
import attr
from . import (
from .exceptions import (
def parse_command_only(self, rawinput: str) -> Statement:
    """Partially parse input into a :class:`~cmd2.Statement` object.

        The command is identified, and shortcuts and aliases are expanded.
        Multiline commands are identified, but terminators and output
        redirection are not parsed.

        This method is used by tab completion code and therefore must not
        generate an exception if there are unclosed quotes.

        The :class:`~cmd2.Statement` object returned by this method can at most
        contain values in the following attributes:
        :attr:`~cmd2.Statement.args`, :attr:`~cmd2.Statement.raw`,
        :attr:`~cmd2.Statement.command`,
        :attr:`~cmd2.Statement.multiline_command`

        :attr:`~cmd2.Statement.args` will include all output redirection
        clauses and command terminators.

        Different from :meth:`~cmd2.parsing.StatementParser.parse` this method
        does not remove redundant whitespace within args. However, it does
        ensure args has no leading or trailing whitespace.

        :param rawinput: the command line as entered by the user
        :return: a new :class:`~cmd2.Statement` object
        """
    line = self._expand(rawinput)
    command = ''
    args = ''
    match = self._command_pattern.search(line)
    if match:
        command = match.group(1)
        args = line[match.end(1):].strip()
        if not command or not args:
            args = ''
    if command in self.multiline_commands:
        multiline_command = command
    else:
        multiline_command = ''
    statement = Statement(args, raw=rawinput, command=command, multiline_command=multiline_command)
    return statement