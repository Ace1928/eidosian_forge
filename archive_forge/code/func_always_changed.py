import json
import os
import sys
import uuid
from collections import defaultdict
from contextlib import contextmanager
from itertools import product
import param
from bokeh.core.property.bases import Property
from bokeh.models import CustomJS
from param.parameterized import Watcher
from ..util import param_watchers
from .model import add_to_doc, diff
from .state import state
@contextmanager
def always_changed(enable):

    def matches(self, new, old):
        return False
    if enable:
        backup = Property.matches
        Property.matches = matches
    try:
        yield
    finally:
        if enable:
            Property.matches = backup