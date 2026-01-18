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
@classmethod
def set_current_backend(cls, backend):
    """Use this method to set the backend to run the switch hooks"""
    for hook in cls._backend_switch_hooks:
        hook(backend)
    cls.current_backend = backend