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
def test_empty_chunked_array():
    msg = 'cannot construct ChunkedArray from empty vector and omitted type'
    with pytest.raises(pa.ArrowInvalid, match=msg):
        pa.chunked_array([])
    pa.chunked_array([], type=pa.int8())