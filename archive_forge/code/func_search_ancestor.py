import re
from codecs import BOM_UTF8
from typing import Tuple
from parso.python.tokenize import group
def search_ancestor(self, *node_types):
    node = self.parent
    while node is not None:
        if node.type in node_types:
            return node
        node = node.parent
    return None