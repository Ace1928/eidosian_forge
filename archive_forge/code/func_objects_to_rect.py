import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def objects_to_rect(objects) -> dict:
    """
    Given an iterable of objects, return the smallest rectangle (i.e. a
    dict with "x0", "top", "x1", and "bottom" keys) that contains them
    all.
    """
    return bbox_to_rect(objects_to_bbox(objects))