from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import instrumentation
from . import interfaces
from . import mapperlib
from .attributes import QueryableAttribute
from .base import _mapper_or_none
from .base import NO_KEY
from .instrumentation import ClassManager
from .instrumentation import InstrumentationFactory
from .query import BulkDelete
from .query import BulkUpdate
from .query import Query
from .scoping import scoped_session
from .session import Session
from .session import sessionmaker
from .. import event
from .. import exc
from .. import util
from ..event import EventTarget
from ..event.registry import _ET
from ..util.compat import inspect_getfullargspec
def after_soft_rollback(self, session: Session, previous_transaction: SessionTransaction) -> None:
    """Execute after any rollback has occurred, including "soft"
        rollbacks that don't actually emit at the DBAPI level.

        This corresponds to both nested and outer rollbacks, i.e.
        the innermost rollback that calls the DBAPI's
        rollback() method, as well as the enclosing rollback
        calls that only pop themselves from the transaction stack.

        The given :class:`.Session` can be used to invoke SQL and
        :meth:`.Session.query` operations after an outermost rollback
        by first checking the :attr:`.Session.is_active` flag::

            @event.listens_for(Session, "after_soft_rollback")
            def do_something(session, previous_transaction):
                if session.is_active:
                    session.execute(text("select * from some_table"))

        :param session: The target :class:`.Session`.
        :param previous_transaction: The :class:`.SessionTransaction`
         transactional marker object which was just closed.   The current
         :class:`.SessionTransaction` for the given :class:`.Session` is
         available via the :attr:`.Session.transaction` attribute.

        """