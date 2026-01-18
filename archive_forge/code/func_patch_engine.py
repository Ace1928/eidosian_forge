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
def patch_engine(self, engine):
    """Patch an Engine into this manager.

        Replaces this manager's factory with a _TestTransactionFactory
        that will use the given Engine, and returns
        a callable that will reset the factory back to what we
        started with.

        Only works for root factories.  Is intended for test suites
        that need to patch in alternate database configurations.

        """
    existing_factory = self._factory
    if not existing_factory._started:
        existing_factory._start()
    maker = existing_factory._writer_maker
    maker_kwargs = existing_factory._maker_args_for_conf(cfg.CONF)
    maker = orm.get_maker(engine=engine, **maker_kwargs)
    factory = _TestTransactionFactory(engine, maker, apply_global=False, from_factory=existing_factory)
    return self.patch_factory(factory)