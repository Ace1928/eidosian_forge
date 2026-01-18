import difflib
import inspect
import pickle
import traceback
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import param
from .accessors import Opts  # noqa (clean up in 2.0)
from .pprint import InfoPrinter
from .tree import AttrTree
from .util import group_sanitizer, label_sanitizer, sanitize_identifier
def format_options_error(self):
    """
        Return a fuzzy match message based on the OptionError
        """
    allowed_keywords = self.allowed_keywords
    target = allowed_keywords.target
    matches = allowed_keywords.fuzzy_match(self.invalid_keyword)
    if not matches:
        matches = allowed_keywords.values
        similarity = 'Possible'
    else:
        similarity = 'Similar'
    loaded_backends = Store.loaded_backends()
    target = f'for {target}' if target else ''
    if len(loaded_backends) == 1:
        loaded = f' in loaded backend {loaded_backends[0]!r}'
    else:
        backend_list = ', '.join([repr(b) for b in loaded_backends[:-1]])
        loaded = f' in loaded backends {backend_list} and {loaded_backends[-1]!r}'
    group = f'{self.group_name} option' if self.group_name else 'keyword'
    return f"Unexpected {group} '{self.invalid_keyword}' {target}{loaded}.\n\n{similarity} keywords in the currently active '{Store.current_backend}' renderer are: {matches}\n\nIf you believe this keyword is correct, please make sure the backend has been imported or loaded with the hv.extension."