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
def patch_factory(self, factory_or_manager):
    """Patch a _TransactionFactory into this manager.

        Replaces this manager's factory with the given one, and returns
        a callable that will reset the factory back to what we
        started with.

        Only works for root factories.  Is intended for test suites
        that need to patch in alternate database configurations.

        The given argument may be a _TransactionContextManager or a
        _TransactionFactory.

        """
    if isinstance(factory_or_manager, _TransactionContextManager):
        factory = factory_or_manager._factory
    elif isinstance(factory_or_manager, _TransactionFactory):
        factory = factory_or_manager
    else:
        raise ValueError('_TransactionContextManager or _TransactionFactory expected.')
    if self._root is not self:
        raise AssertionError('patch_factory only works for root factory.')
    existing_factory = self._root_factory
    self._root_factory = factory

    def reset():
        self._root_factory = existing_factory
    return reset