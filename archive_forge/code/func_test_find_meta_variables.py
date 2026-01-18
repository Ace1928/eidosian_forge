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
def test_find_meta_variables(self):
    """Test :func:`humanfriendly.usage.find_meta_variables()`."""
    assert sorted(find_meta_variables("\n            Here's one example: --format-number=VALUE\n            Here's another example: --format-size=BYTES\n            A final example: --format-timespan=SECONDS\n            This line doesn't contain a META variable.\n        ")) == sorted(['VALUE', 'BYTES', 'SECONDS'])