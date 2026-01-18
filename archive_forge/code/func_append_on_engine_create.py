import contextlib
import functools
import inspect
import operator
import threading
import warnings
import debtcollector.moves
import debtcollector.removals
import debtcollector.renames
from oslo_config import cfg
from oslo_utils import excutils
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import orm
from oslo_db import warning
def append_on_engine_create(self, fn):
    """Append a listener function to _facade_cfg["on_engine_create"]"""
    self._factory._facade_cfg['on_engine_create'].append(fn)