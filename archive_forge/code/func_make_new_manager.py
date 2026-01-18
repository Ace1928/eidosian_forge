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
def make_new_manager(self):
    """Create a new, independent _TransactionContextManager from this one.

        Copies the underlying _TransactionFactory to a new one, so that
        it can be further configured with new options.

        Used for test environments where the application-wide
        _TransactionContextManager may be used as a factory for test-local
        managers.

        """
    new = self._clone()
    new._root = new
    new._root_factory = self._root_factory._create_factory_copy()
    if new._factory._started:
        raise AssertionError('TransactionFactory is already started')
    return new