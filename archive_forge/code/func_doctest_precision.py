import gc
import io
import os
import re
import shlex
import sys
import warnings
from importlib import invalidate_caches
from io import StringIO
from pathlib import Path
from textwrap import dedent
from unittest import TestCase, mock
import pytest
from IPython import get_ipython
from IPython.core import magic
from IPython.core.error import UsageError
from IPython.core.magic import (
from IPython.core.magics import code, execution, logging, osm, script
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
from IPython.utils.process import find_cmd
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.syspathcontext import prepended_to_syspath
from .test_debugger import PdbTestInput
from tempfile import NamedTemporaryFile
from IPython.core.magic import (
def doctest_precision():
    """doctest for %precision

    In [1]: f = get_ipython().display_formatter.formatters['text/plain']

    In [2]: %precision 5
    Out[2]: '%.5f'

    In [3]: f.float_format
    Out[3]: '%.5f'

    In [4]: %precision %e
    Out[4]: '%e'

    In [5]: f(3.1415927)
    Out[5]: '3.141593e+00'
    """