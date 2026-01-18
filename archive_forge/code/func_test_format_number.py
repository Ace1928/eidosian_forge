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
def test_format_number(self):
    """Test :func:`humanfriendly.format_number()`."""
    self.assertEqual('1', format_number(1))
    self.assertEqual('1.5', format_number(1.5))
    self.assertEqual('1.56', format_number(1.56789))
    self.assertEqual('1.567', format_number(1.56789, 3))
    self.assertEqual('1,000', format_number(1000))
    self.assertEqual('1,000', format_number(1000.12, 0))
    self.assertEqual('1,000,000', format_number(1000000))
    self.assertEqual('1,000,000.42', format_number(1000000.42))
    self.assertEqual('-285.67', format_number(-285.67))