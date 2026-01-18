import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def enabled_for_config(self, config):
    for predicate in self.skips.union(self.fails):
        if predicate(config):
            return False
    else:
        return True