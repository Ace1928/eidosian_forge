import logging
import optparse
import os
import re
import shlex
import urllib.parse
from optparse import Values
from typing import (
from pip._internal.cli import cmdoptions
from pip._internal.exceptions import InstallationError, RequirementsFileParseError
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.encoding import auto_decode
from pip._internal.utils.urls import get_url_scheme
def join_lines(lines_enum: ReqFileLines) -> ReqFileLines:
    """Joins a line ending in '' with the previous line (except when following
    comments).  The joined line takes on the index of the first line.
    """
    primary_line_number = None
    new_line: List[str] = []
    for line_number, line in lines_enum:
        if not line.endswith('\\') or COMMENT_RE.match(line):
            if COMMENT_RE.match(line):
                line = ' ' + line
            if new_line:
                new_line.append(line)
                assert primary_line_number is not None
                yield (primary_line_number, ''.join(new_line))
                new_line = []
            else:
                yield (line_number, line)
        else:
            if not new_line:
                primary_line_number = line_number
            new_line.append(line.strip('\\'))
    if new_line:
        assert primary_line_number is not None
        yield (primary_line_number, ''.join(new_line))