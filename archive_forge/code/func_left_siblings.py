import warnings
from abc import ABCMeta, abstractmethod
from nltk.tree.tree import Tree
from nltk.util import slice_bounds
def left_siblings(self):
    """
        A list of all left siblings of this tree, in any of its parent
        trees.  A tree may be its own left sibling if it is used as
        multiple contiguous children of the same parent.  A tree may
        appear multiple times in this list if it is the left sibling
        of this tree with respect to multiple parents.

        :type: list(MultiParentedTree)
        """
    return [parent[index - 1] for parent, index in self._get_parent_indices() if index > 0]