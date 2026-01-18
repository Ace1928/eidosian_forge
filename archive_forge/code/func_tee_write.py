from io import StringIO
import tempfile
from typing import IO
from typing import Union
from _pytest.config import Config
from _pytest.config import create_terminal_writer
from _pytest.config.argparsing import Parser
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
import pytest
def tee_write(s, **kwargs):
    oldwrite(s, **kwargs)
    if isinstance(s, str):
        s = s.encode('utf-8')
    config.stash[pastebinfile_key].write(s)