import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def db_spec(*dbs):
    return OrPredicate([Predicate.as_predicate(db) for db in dbs])