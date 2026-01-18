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
@property
def allow_async(self):
    """Modifier to allow async operations

        Allows async operations if asynchronous session is already
        started in this context. Marking DB API methods with READER would make
        it impossible to use them in ASYNC_READER transactions, and marking
        them with ASYNC_READER would require a modification of all the places
        these DB API methods are called to force READER mode, where the latest
        DB state is required.

        In Nova DB API methods should have a 'safe' default (i.e. READER),
        so that they can start sessions on their own, but it would also be
        useful for them to be able to participate in an existing ASYNC_READER
        session, if one was started up the stack.
        """
    if self._mode is _WRITER:
        raise TypeError('Setting async on a WRITER makes no sense')
    return self._clone(allow_async=True)