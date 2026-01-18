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
def start_recording_skipped(cls):
    """
        Start collecting OptionErrors for all skipped options recorded
        with the record_skipped_option method
        """
    cls._errors_recorded = []