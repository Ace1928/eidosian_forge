import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def read_to_next_empty_line(self):
    self.seek_next_non_empty_line()

    def is_empty(line):
        return not line.strip()
    return self.read_to_condition(is_empty)