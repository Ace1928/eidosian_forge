import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def get_error_node(self, node):
    return node.parent