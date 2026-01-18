import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def edges_to_set(edges):
    return set(map(obj_to_bbox, edges))