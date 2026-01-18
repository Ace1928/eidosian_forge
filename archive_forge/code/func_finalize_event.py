import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
@finalize_event.setter
def finalize_event(self, value):
    self._finalize_event = listify(value)