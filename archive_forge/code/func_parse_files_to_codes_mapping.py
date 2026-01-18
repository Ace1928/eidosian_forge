from __future__ import annotations
import fnmatch as _fnmatch
import functools
import io
import logging
import os
import platform
import re
import sys
import textwrap
import tokenize
from typing import NamedTuple
from typing import Pattern
from typing import Sequence
from flake8 import exceptions
def parse_files_to_codes_mapping(value_: Sequence[str] | str) -> list[tuple[str, list[str]]]:
    """Parse a files-to-codes mapping.

    A files-to-codes mapping a sequence of values specified as
    `filenames list:codes list ...`.  Each of the lists may be separated by
    either comma or whitespace tokens.

    :param value: String to be parsed and normalized.
    """
    if not isinstance(value_, str):
        value = '\n'.join(value_)
    else:
        value = value_
    ret: list[tuple[str, list[str]]] = []
    if not value.strip():
        return ret

    class State:
        seen_sep = True
        seen_colon = False
        filenames: list[str] = []
        codes: list[str] = []

    def _reset() -> None:
        if State.codes:
            for filename in State.filenames:
                ret.append((filename, State.codes))
        State.seen_sep = True
        State.seen_colon = False
        State.filenames = []
        State.codes = []

    def _unexpected_token() -> exceptions.ExecutionError:
        return exceptions.ExecutionError(f'Expected `per-file-ignores` to be a mapping from file exclude patterns to ignore codes.\n\nConfigured `per-file-ignores` setting:\n\n{textwrap.indent(value.strip(), '    ')}')
    for token in _tokenize_files_to_codes_mapping(value):
        if token.tp in {_COMMA, _WS}:
            State.seen_sep = True
        elif not State.seen_colon:
            if token.tp == _COLON:
                State.seen_colon = True
                State.seen_sep = True
            elif State.seen_sep and token.tp == _FILE:
                State.filenames.append(token.src)
                State.seen_sep = False
            else:
                raise _unexpected_token()
        elif token.tp == _EOF:
            _reset()
        elif State.seen_sep and token.tp == _CODE:
            State.codes.append(token.src)
            State.seen_sep = False
        elif State.seen_sep and token.tp == _FILE:
            _reset()
            State.filenames.append(token.src)
            State.seen_sep = False
        else:
            raise _unexpected_token()
    return ret