import collections
import doctest
import types
from typing import Any, Iterator, Mapping
import unittest
from absl.testing import parameterized
import attr
import numpy as np
import tree
import wrapt
def get_paths_and_values(shallow_tree, input_tree):
    path_value_pairs = tree.flatten_with_path_up_to(shallow_tree, input_tree)
    paths = [p for p, _ in path_value_pairs]
    values = [v for _, v in path_value_pairs]
    return (paths, values)