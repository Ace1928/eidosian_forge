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
@_lifecycle_event
def persistent_to_deleted(self, session: Session, instance: _O) -> None:
    """Intercept the "persistent to deleted" transition for a specific
        object.

        This event is invoked when a persistent object's identity
        is deleted from the database within a flush, however the object
        still remains associated with the :class:`.Session` until the
        transaction completes.

        If the transaction is rolled back, the object moves again
        to the persistent state, and the
        :meth:`.SessionEvents.deleted_to_persistent` event is called.
        If the transaction is committed, the object becomes detached,
        which will emit the :meth:`.SessionEvents.deleted_to_detached`
        event.

        Note that while the :meth:`.Session.delete` method is the primary
        public interface to mark an object as deleted, many objects
        get deleted due to cascade rules, which are not always determined
        until flush time.  Therefore, there's no way to catch
        every object that will be deleted until the flush has proceeded.
        the :meth:`.SessionEvents.persistent_to_deleted` event is therefore
        invoked at the end of a flush.

        .. seealso::

            :ref:`session_lifecycle_events`

        """