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
def after_begin(self, session: Session, transaction: SessionTransaction, connection: Connection) -> None:
    """Execute after a transaction is begun on a connection.

        .. note:: This event is called within the process of the
          :class:`_orm.Session` modifying its own internal state.
          To invoke SQL operations within this hook, use the
          :class:`_engine.Connection` provided to the event;
          do not run SQL operations using the :class:`_orm.Session`
          directly.

        :param session: The target :class:`.Session`.
        :param transaction: The :class:`.SessionTransaction`.
        :param connection: The :class:`_engine.Connection` object
         which will be used for SQL statements.

        .. seealso::

            :meth:`~.SessionEvents.before_commit`

            :meth:`~.SessionEvents.after_commit`

            :meth:`~.SessionEvents.after_transaction_create`

            :meth:`~.SessionEvents.after_transaction_end`

        """