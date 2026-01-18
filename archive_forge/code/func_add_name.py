import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def add_name(self, name):
    parent_type = name.parent.type
    if parent_type == 'trailer':
        return
    if parent_type == 'global_stmt':
        self._global_names.append(name)
    elif parent_type == 'nonlocal_stmt':
        self._nonlocal_names.append(name)
    elif parent_type == 'funcdef':
        self._local_params_names.extend([param.name.value for param in name.parent.get_params()])
    else:
        self._used_name_dict.setdefault(name.value, []).append(name)