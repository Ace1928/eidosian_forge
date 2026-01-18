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
def assert_row(self, class_, rowobj, desc):
    self.assert_(rowobj.__class__ is class_, 'item class is not ' + repr(class_))
    for key, value in desc.items():
        if isinstance(value, tuple):
            if isinstance(value[1], list):
                self.assert_list(getattr(rowobj, key), value[0], value[1])
            else:
                self.assert_row(value[0], getattr(rowobj, key), value[1])
        else:
            self.assert_(getattr(rowobj, key) == value, 'attribute %s value %s does not match %s' % (key, getattr(rowobj, key), value))