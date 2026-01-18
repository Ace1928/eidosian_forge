import datetime
import math
import os
import random
import re
import subprocess
import sys
import time
import types
import unittest
import warnings
from humanfriendly import (
from humanfriendly.case import CaseInsensitiveDict, CaseInsensitiveKey
from humanfriendly.cli import main
from humanfriendly.compat import StringIO
from humanfriendly.decorators import cached
from humanfriendly.deprecation import DeprecationProxy, define_aliases, deprecated_args, get_aliases
from humanfriendly.prompts import (
from humanfriendly.sphinx import (
from humanfriendly.tables import (
from humanfriendly.terminal import (
from humanfriendly.terminal.html import html_to_ansi
from humanfriendly.terminal.spinners import AutomaticSpinner, Spinner
from humanfriendly.testing import (
from humanfriendly.text import (
from humanfriendly.usage import (
from mock import MagicMock
def test_find_terminal_size(self):
    """Test :func:`humanfriendly.terminal.find_terminal_size()`."""
    lines, columns = find_terminal_size()
    assert lines > 0
    assert columns > 0
    saved_stdin = sys.stdin
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    try:
        sys.stdin = StringIO()
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        lines, columns = find_terminal_size()
        assert lines > 0
        assert columns > 0
        saved_path = os.environ['PATH']
        try:
            os.environ['PATH'] = ''
            lines, columns = find_terminal_size()
            assert lines > 0
            assert columns > 0
        finally:
            os.environ['PATH'] = saved_path
    finally:
        sys.stdin = saved_stdin
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr