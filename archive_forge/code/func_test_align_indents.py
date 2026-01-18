import json
from textwrap import dedent, indent
from unittest.mock import Mock, patch
import numpy as np
import pandas
import pytest
import modin.pandas as pd
import modin.utils
from modin.error_message import ErrorMessage
from modin.tests.pandas.utils import create_test_dfs
def test_align_indents():
    source = '\n    Source string that sets\n        the indent pattern.'
    target = indent(source, ' ' * 5)
    result = modin.utils.align_indents(source, target)
    assert source == result