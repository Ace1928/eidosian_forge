import collections
from functools import partial
import string
import subprocess
import sys
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
from pandas.core import ops
import pandas.core.common as com
from pandas.util.version import Version
def test_git_version():
    git_version = pd.__git_version__
    assert len(git_version) == 40
    assert all((c in string.hexdigits for c in git_version))