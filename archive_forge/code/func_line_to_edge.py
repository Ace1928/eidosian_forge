import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def line_to_edge(line):
    edge = dict(line)
    edge['orientation'] = 'h' if line['top'] == line['bottom'] else 'v'
    return edge