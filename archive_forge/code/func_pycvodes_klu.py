from __future__ import (absolute_import, division, print_function)
from functools import reduce
import inspect
import math
import operator
import sys
from pkg_resources import parse_requirements, parse_version
import numpy as np
import pytest
def pycvodes_klu(cb):
    try:
        from pycvodes import config
        klu = config['KLU']
    except (ModuleNotFoundError, ImportError):
        klu = False
    return pytest.mark.skipif(not klu, reason='Sparse jacobian tests require pycvodes and sundials with KLU enabled.')(cb)