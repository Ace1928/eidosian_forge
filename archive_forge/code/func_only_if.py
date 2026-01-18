import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def only_if(predicate, reason=None):
    predicate = _as_predicate(predicate)
    return skip_if(NotPredicate(predicate), reason)