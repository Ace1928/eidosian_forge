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
def after_commit(self, session: Session) -> None:
    """Execute after a commit has occurred.

        .. note::

            The :meth:`~.SessionEvents.after_commit` hook is *not* per-flush,
            that is, the :class:`.Session` can emit SQL to the database
            many times within the scope of a transaction.
            For interception of these events, use the
            :meth:`~.SessionEvents.before_flush`,
            :meth:`~.SessionEvents.after_flush`, or
            :meth:`~.SessionEvents.after_flush_postexec`
            events.

        .. note::

            The :class:`.Session` is not in an active transaction
            when the :meth:`~.SessionEvents.after_commit` event is invoked,
            and therefore can not emit SQL.  To emit SQL corresponding to
            every transaction, use the :meth:`~.SessionEvents.before_commit`
            event.

        :param session: The target :class:`.Session`.

        .. seealso::

            :meth:`~.SessionEvents.before_commit`

            :meth:`~.SessionEvents.after_begin`

            :meth:`~.SessionEvents.after_transaction_create`

            :meth:`~.SessionEvents.after_transaction_end`

        """