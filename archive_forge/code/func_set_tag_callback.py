import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def set_tag_callback(cbk):
    """
    Every stat. entry will have a specific tag field and users might be able
    to filter on stats via tag field.
    """
    return _yappi.set_tag_callback(cbk)