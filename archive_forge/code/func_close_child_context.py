import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def close_child_context(self, child_context):
    self._nonlocal_names_in_subscopes += child_context.finalize()