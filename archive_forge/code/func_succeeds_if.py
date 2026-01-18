import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def succeeds_if(predicate, reason=None):
    predicate = _as_predicate(predicate)
    return fails_if(NotPredicate(predicate), reason)