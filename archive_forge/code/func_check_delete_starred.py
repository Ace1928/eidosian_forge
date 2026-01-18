import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def check_delete_starred(node):
    while node.parent is not None:
        node = node.parent
        if node.type == 'del_stmt':
            return True
        if node.type not in (*_STAR_EXPR_PARENTS, 'atom'):
            return False
    return False