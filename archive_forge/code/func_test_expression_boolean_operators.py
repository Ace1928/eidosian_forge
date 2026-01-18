from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
def test_expression_boolean_operators():
    true = pc.scalar(True)
    false = pc.scalar(False)
    with pytest.raises(ValueError, match='cannot be evaluated to python True'):
        true and false
    with pytest.raises(ValueError, match='cannot be evaluated to python True'):
        true or false
    with pytest.raises(ValueError, match='cannot be evaluated to python True'):
        bool(true)
    with pytest.raises(ValueError, match='cannot be evaluated to python True'):
        not true