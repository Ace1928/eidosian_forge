from __future__ import annotations
from collections import deque
import decimal
import gc
from itertools import chain
import random
import sys
from sys import getsizeof
import types
from . import config
from . import mock
from .. import inspect
from ..engine import Connection
from ..schema import Column
from ..schema import DropConstraint
from ..schema import DropTable
from ..schema import ForeignKeyConstraint
from ..schema import MetaData
from ..schema import Table
from ..sql import schema
from ..sql.sqltypes import Integer
from ..util import decorator
from ..util import defaultdict
from ..util import has_refcount_gc
from ..util import inspect_getfullargspec
def resolve_lambda(__fn, **kw):
    """Given a no-arg lambda and a namespace, return a new lambda that
    has all the values filled in.

    This is used so that we can have module-level fixtures that
    refer to instance-level variables using lambdas.

    """
    pos_args = inspect_getfullargspec(__fn)[0]
    pass_pos_args = {arg: kw.pop(arg) for arg in pos_args}
    glb = dict(__fn.__globals__)
    glb.update(kw)
    new_fn = types.FunctionType(__fn.__code__, glb)
    return new_fn(**pass_pos_args)