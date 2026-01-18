import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def matching_config_reasons(self, config):
    return [predicate._as_string(config) for predicate in self.skips.union(self.fails) if predicate(config)]