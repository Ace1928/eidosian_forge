import re
import shlex
from typing import (
import attr
from . import (
from .exceptions import (
def is_valid_command(self, word: str, *, is_subcommand: bool=False) -> Tuple[bool, str]:
    """Determine whether a word is a valid name for a command.

        Commands cannot include redirection characters, whitespace,
        or termination characters. They also cannot start with a
        shortcut.

        :param word: the word to check as a command
        :param is_subcommand: Flag whether this command name is a subcommand name
        :return: a tuple of a boolean and an error string

        If word is not a valid command, return ``False`` and an error string
        suitable for inclusion in an error message of your choice::

            checkit = '>'
            valid, errmsg = statement_parser.is_valid_command(checkit)
            if not valid:
                errmsg = f"alias: {errmsg}"
        """
    valid = False
    if not isinstance(word, str):
        return (False, f'must be a string. Received {str(type(word))} instead')
    if not word:
        return (False, 'cannot be an empty string')
    if word.startswith(constants.COMMENT_CHAR):
        return (False, 'cannot start with the comment character')
    if not is_subcommand:
        for shortcut, _ in self.shortcuts:
            if word.startswith(shortcut):
                errmsg = 'cannot start with a shortcut: '
                errmsg += ', '.join((shortcut for shortcut, _ in self.shortcuts))
                return (False, errmsg)
    errmsg = 'cannot contain: whitespace, quotes, '
    errchars = []
    errchars.extend(constants.REDIRECTION_CHARS)
    errchars.extend(self.terminators)
    errmsg += ', '.join([shlex.quote(x) for x in errchars])
    match = self._command_pattern.search(word)
    if match:
        if word == match.group(1):
            valid = True
            errmsg = ''
    return (valid, errmsg)