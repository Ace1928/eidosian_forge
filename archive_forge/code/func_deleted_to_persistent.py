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
def deleted_to_persistent(self, session: Session, instance: _O) -> None:
    """Intercept the "deleted to persistent" transition for a specific
        object.

        This transition occurs only when an object that's been deleted
        successfully in a flush is restored due to a call to
        :meth:`.Session.rollback`.   The event is not called under
        any other circumstances.

        .. seealso::

            :ref:`session_lifecycle_events`

        """