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
def lambda_combinations(lambda_arg_sets, **kw):
    args = inspect_getfullargspec(lambda_arg_sets)
    arg_sets = lambda_arg_sets(*[mock.Mock() for arg in args[0]])

    def create_fixture(pos):

        def fixture(**kw):
            return lambda_arg_sets(**kw)[pos]
        fixture.__name__ = 'fixture_%3.3d' % pos
        return fixture
    return config.combinations(*[(create_fixture(i),) for i in range(len(arg_sets))], **kw)