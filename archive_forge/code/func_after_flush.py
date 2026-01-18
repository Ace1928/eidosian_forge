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
def after_flush(self, session: Session, flush_context: UOWTransaction) -> None:
    """Execute after flush has completed, but before commit has been
        called.

        Note that the session's state is still in pre-flush, i.e. 'new',
        'dirty', and 'deleted' lists still show pre-flush state as well
        as the history settings on instance attributes.

        .. warning:: This event runs after the :class:`.Session` has emitted
           SQL to modify the database, but **before** it has altered its
           internal state to reflect those changes, including that newly
           inserted objects are placed into the identity map.  ORM operations
           emitted within this event such as loads of related items
           may produce new identity map entries that will immediately
           be replaced, sometimes causing confusing results.  SQLAlchemy will
           emit a warning for this condition as of version 1.3.9.

        :param session: The target :class:`.Session`.
        :param flush_context: Internal :class:`.UOWTransaction` object
         which handles the details of the flush.

        .. seealso::

            :meth:`~.SessionEvents.before_flush`

            :meth:`~.SessionEvents.after_flush_postexec`

            :ref:`session_persistence_events`

        """