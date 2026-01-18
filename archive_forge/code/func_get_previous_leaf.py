from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union
from parso.utils import split_lines
def get_previous_leaf(self):
    """
        Returns the previous leaf in the parser tree.
        Returns `None` if this is the first element in the parser tree.
        """
    if self.parent is None:
        return None
    node = self
    while True:
        c = node.parent.children
        i = c.index(node)
        if i == 0:
            node = node.parent
            if node.parent is None:
                return None
        else:
            node = c[i - 1]
            break
    while True:
        try:
            node = node.children[-1]
        except AttributeError:
            return node