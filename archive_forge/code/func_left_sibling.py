import warnings
from abc import ABCMeta, abstractmethod
from nltk.tree.tree import Tree
from nltk.util import slice_bounds
def left_sibling(self):
    """The left sibling of this tree, or None if it has none."""
    parent_index = self.parent_index()
    if self._parent and parent_index > 0:
        return self._parent[parent_index - 1]
    return None