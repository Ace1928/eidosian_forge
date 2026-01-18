from __future__ import annotations
from collections import defaultdict
import contextlib
from copy import copy
from itertools import filterfalse
import re
import sys
import warnings
from . import assertsql
from . import config
from . import engines
from . import mock
from .exclusions import db_spec
from .util import fail
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import types as sqltypes
from .. import util
from ..engine import default
from ..engine import url
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import decorator
def emits_warning(*messages):
    """Decorator form of expect_warnings().

    Note that emits_warning does **not** assert that the warnings
    were in fact seen.

    """

    @decorator
    def decorate(fn, *args, **kw):
        with expect_warnings(*messages, assert_=False):
            return fn(*args, **kw)
    return decorate