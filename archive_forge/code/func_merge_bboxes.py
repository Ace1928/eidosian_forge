import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def merge_bboxes(bboxes):
    """
    Given an iterable of bounding boxes, return the smallest bounding box
    that contains them all.
    """
    x0, top, x1, bottom = zip(*bboxes)
    return (min(x0), min(top), max(x1), max(bottom))