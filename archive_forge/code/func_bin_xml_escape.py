from datetime import datetime
import functools
import os
import platform
import re
from typing import Callable
from typing import Dict
from typing import List
from typing import Match
from typing import Optional
from typing import Tuple
from typing import Union
import xml.etree.ElementTree as ET
from _pytest import nodes
from _pytest import timing
from _pytest._code.code import ExceptionRepr
from _pytest._code.code import ReprFileLocation
from _pytest.config import Config
from _pytest.config import filename_arg
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.reports import TestReport
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
import pytest
def bin_xml_escape(arg: object) -> str:
    """Visually escape invalid XML characters.

    For example, transforms
        'hello\\aworld\\b'
    into
        'hello#x07world#x08'
    Note that the #xABs are *not* XML escapes - missing the ampersand &#xAB.
    The idea is to escape visually for the user rather than for XML itself.
    """

    def repl(matchobj: Match[str]) -> str:
        i = ord(matchobj.group())
        if i <= 255:
            return '#x%02X' % i
        else:
            return '#x%04X' % i
    illegal_xml_re = '[^\t\n\r -~\x80-\ud7ff\ue000-�က0-ჿFF]'
    return re.sub(illegal_xml_re, repl, str(arg))