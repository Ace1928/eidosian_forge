import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
@prepare_event.setter
def prepare_event(self, value):
    self._prepare_event = listify(value)