from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc
import numpy as np
import pytest
from pandas._libs import hashtable as ht
import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin
def get_allocated_khash_memory():
    snapshot = tracemalloc.take_snapshot()
    snapshot = snapshot.filter_traces((tracemalloc.DomainFilter(True, ht.get_hashtable_trace_domain()),))
    return sum((x.size for x in snapshot.traces))