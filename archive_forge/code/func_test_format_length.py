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
def test_format_length(self):
    """Test :func:`humanfriendly.format_length()`."""
    self.assertEqual('0 metres', format_length(0))
    self.assertEqual('1 metre', format_length(1))
    self.assertEqual('42 metres', format_length(42))
    self.assertEqual('1 km', format_length(1 * 1000))
    self.assertEqual('15.3 cm', format_length(0.153))
    self.assertEqual('1 cm', format_length(0.01))
    self.assertEqual('1 mm', format_length(0.001))
    self.assertEqual('1 nm', format_length(1e-09))