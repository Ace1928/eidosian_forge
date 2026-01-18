important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def get_corresponding_test_node(self, node):
    """
        Searches for the branch in which the node is and returns the
        corresponding test node (see function above). However if the node is in
        the test node itself and not in the suite return None.
        """
    start_pos = node.start_pos
    for check_node in reversed(list(self.get_test_nodes())):
        if check_node.start_pos < start_pos:
            if start_pos < check_node.end_pos:
                return None
            else:
                return check_node