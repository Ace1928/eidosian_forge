from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union
from parso.utils import split_lines
def get_leaf_for_position(self, position, include_prefixes=False):
    """
        Get the :py:class:`parso.tree.Leaf` at ``position``

        :param tuple position: A position tuple, row, column. Rows start from 1
        :param bool include_prefixes: If ``False``, ``None`` will be returned if ``position`` falls
            on whitespace or comments before a leaf
        :return: :py:class:`parso.tree.Leaf` at ``position``, or ``None``
        """

    def binary_search(lower, upper):
        if lower == upper:
            element = self.children[lower]
            if not include_prefixes and position < element.start_pos:
                return None
            try:
                return element.get_leaf_for_position(position, include_prefixes)
            except AttributeError:
                return element
        index = int((lower + upper) / 2)
        element = self.children[index]
        if position <= element.end_pos:
            return binary_search(lower, index)
        else:
            return binary_search(index + 1, upper)
    if not (1, 0) <= position <= self.children[-1].end_pos:
        raise ValueError('Please provide a position that exists within this node.')
    return binary_search(0, len(self.children) - 1)