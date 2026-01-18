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
def persistent_to_detached(self, session: Session, instance: _O) -> None:
    """Intercept the "persistent to detached" transition for a specific
        object.

        This event is invoked when a persistent object is evicted
        from the session.  There are many conditions that cause this
        to happen, including:

        * using a method such as :meth:`.Session.expunge`
          or :meth:`.Session.close`

        * Calling the :meth:`.Session.rollback` method, when the object
          was part of an INSERT statement for that session's transaction


        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        :param deleted: boolean.  If True, indicates this object moved
         to the detached state because it was marked as deleted and flushed.


        .. seealso::

            :ref:`session_lifecycle_events`

        """