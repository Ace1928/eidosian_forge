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
def flag_combinations(*combinations):
    """A facade around @testing.combinations() oriented towards boolean
    keyword-based arguments.

    Basically generates a nice looking identifier based on the keywords
    and also sets up the argument names.

    E.g.::

        @testing.flag_combinations(
            dict(lazy=False, passive=False),
            dict(lazy=True, passive=False),
            dict(lazy=False, passive=True),
            dict(lazy=False, passive=True, raiseload=True),
        )


    would result in::

        @testing.combinations(
            ('', False, False, False),
            ('lazy', True, False, False),
            ('lazy_passive', True, True, False),
            ('lazy_passive', True, True, True),
            id_='iaaa',
            argnames='lazy,passive,raiseload'
        )

    """
    keys = set()
    for d in combinations:
        keys.update(d)
    keys = sorted(keys)
    return config.combinations(*[('_'.join((k for k in keys if d.get(k, False))),) + tuple((d.get(k, False) for k in keys)) for d in combinations], id_='i' + 'a' * len(keys), argnames=','.join(keys))