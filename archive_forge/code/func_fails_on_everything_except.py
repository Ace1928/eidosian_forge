import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def fails_on_everything_except(*dbs):
    return succeeds_if(OrPredicate([Predicate.as_predicate(db) for db in dbs]))