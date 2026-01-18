import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
def get_latest_suite_node(self):
    n = self
    while n is not None:
        if n.type == IndentationTypes.SUITE:
            return n
        n = n.parent