import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def make_cluster_dict(values, tolerance) -> dict:
    clusters = cluster_list(list(set(values)), tolerance)
    nested_tuples = [[(val, i) for val in value_cluster] for i, value_cluster in enumerate(clusters)]
    return dict(itertools.chain(*nested_tuples))