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
def get_sessionmaker(self, use_slave=False):
    """Get the sessionmaker instance used to create a Session.

        This can be called for those cases where the sessionmaker() is to
        be temporarily injected with some state such as a specific connection.

        """
    self._check_factory_started()
    if use_slave:
        return self._factory._reader_maker
    else:
        return self._factory._writer_maker