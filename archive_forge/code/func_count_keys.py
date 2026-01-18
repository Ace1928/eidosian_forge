import json
import sys
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterable, Iterator
import ase.io
from ase.db import connect
from ase.db.core import convert_str_to_int_float_or_str
from ase.db.row import row2dct
from ase.db.table import Table, all_columns
from ase.utils import plural
def count_keys(db, query):
    keys = defaultdict(int)
    for row in db.select(query):
        for key in row._keys:
            keys[key] += 1
    n = max((len(key) for key in keys)) + 1
    for key, number in keys.items():
        print('{:{}} {}'.format(key + ':', n, number))
    return