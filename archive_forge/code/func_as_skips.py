import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def as_skips(self):
    rule = compound()
    rule.skips.update(self.skips)
    rule.skips.update(self.fails)
    return rule