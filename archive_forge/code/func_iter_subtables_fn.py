from collections import deque
from typing import Callable, Deque, Iterable, List, Optional, Tuple
from .otBase import BaseTable
def iter_subtables_fn(table):
    return table.iterSubTables()