import sys
from copy import deepcopy
from typing import List, Callable, Iterator, Union, Optional, Generic, TypeVar, TYPE_CHECKING
from collections import OrderedDict
def expand_kids_by_data(self, *data_values):
    """Expand (inline) children with any of the given data values. Returns True if anything changed"""
    changed = False
    for i in range(len(self.children) - 1, -1, -1):
        child = self.children[i]
        if isinstance(child, Tree) and child.data in data_values:
            self.children[i:i + 1] = child.children
            changed = True
    return changed