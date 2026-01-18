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
def assert_warns_message(except_cls, msg, callable_, *args, **kwargs):
    """legacy adapter function for functions that were previously using
    assert_raises with SAWarning or similar.

    has some workarounds to accommodate the fact that the callable completes
    with this approach rather than stopping at the exception raise.

    Also uses regex.search() to match the given message to the error string
    rather than regex.match().

    """
    with _expect_warnings_sqla_only(except_cls, [msg], search_msg=True, regex=False):
        return callable_(*args, **kwargs)