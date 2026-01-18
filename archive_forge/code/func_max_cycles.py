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
def max_cycles(self, num):
    """
        Truncates all contained Palette objects to a maximum number
        of samples and returns a new Options object containing the
        truncated or resampled Palettes.
        """
    kwargs = {kw: arg[num] if isinstance(arg, Palette) else arg for kw, arg in self.kwargs.items()}
    return self(max_cycles=num, **kwargs)