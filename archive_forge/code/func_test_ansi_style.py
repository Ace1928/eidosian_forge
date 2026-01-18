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
def test_ansi_style(self):
    """Test :func:`humanfriendly.terminal.ansi_style()`."""
    assert ansi_style(bold=True) == '%s1%s' % (ANSI_CSI, ANSI_SGR)
    assert ansi_style(faint=True) == '%s2%s' % (ANSI_CSI, ANSI_SGR)
    assert ansi_style(italic=True) == '%s3%s' % (ANSI_CSI, ANSI_SGR)
    assert ansi_style(underline=True) == '%s4%s' % (ANSI_CSI, ANSI_SGR)
    assert ansi_style(inverse=True) == '%s7%s' % (ANSI_CSI, ANSI_SGR)
    assert ansi_style(strike_through=True) == '%s9%s' % (ANSI_CSI, ANSI_SGR)
    assert ansi_style(color='blue') == '%s34%s' % (ANSI_CSI, ANSI_SGR)
    assert ansi_style(background='blue') == '%s44%s' % (ANSI_CSI, ANSI_SGR)
    assert ansi_style(color='blue', bright=True) == '%s94%s' % (ANSI_CSI, ANSI_SGR)
    assert ansi_style(color=214) == '%s38;5;214%s' % (ANSI_CSI, ANSI_SGR)
    assert ansi_style(background=214) == '%s39;5;214%s' % (ANSI_CSI, ANSI_SGR)
    assert ansi_style(color=(0, 0, 0)) == '%s38;2;0;0;0%s' % (ANSI_CSI, ANSI_SGR)
    assert ansi_style(color=(255, 255, 255)) == '%s38;2;255;255;255%s' % (ANSI_CSI, ANSI_SGR)
    assert ansi_style(background=(50, 100, 150)) == '%s48;2;50;100;150%s' % (ANSI_CSI, ANSI_SGR)
    with self.assertRaises(ValueError):
        ansi_style(color='unknown')