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
def eq_ignore_whitespace(a, b, msg=None):
    a = re.sub('^\\s+?|\\n', '', a)
    a = re.sub(' {2,}', ' ', a)
    a = re.sub('\\t', '', a)
    b = re.sub('^\\s+?|\\n', '', b)
    b = re.sub(' {2,}', ' ', b)
    b = re.sub('\\t', '', b)
    assert a == b, msg or '%r != %r' % (a, b)