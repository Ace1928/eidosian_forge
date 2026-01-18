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
def test_exported_option_classes():
    classes = exported_option_classes
    assert len(classes) >= 10
    for cls in classes:
        sig = inspect.signature(cls)
        for param in sig.parameters.values():
            assert param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)