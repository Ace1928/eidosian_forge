import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def search_all_comp_ancestors(node):
    has_ancestors = False
    while True:
        node = node.search_ancestor('testlist_comp', 'dictorsetmaker')
        if node is None:
            break
        for child in node.children:
            if child.type in _COMP_FOR_TYPES:
                process_comp_for(child)
                has_ancestors = True
                break
    return has_ancestors