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
def after_rollback(self, session: Session) -> None:
    """Execute after a real DBAPI rollback has occurred.

        Note that this event only fires when the *actual* rollback against
        the database occurs - it does *not* fire each time the
        :meth:`.Session.rollback` method is called, if the underlying
        DBAPI transaction has already been rolled back.  In many
        cases, the :class:`.Session` will not be in
        an "active" state during this event, as the current
        transaction is not valid.   To acquire a :class:`.Session`
        which is active after the outermost rollback has proceeded,
        use the :meth:`.SessionEvents.after_soft_rollback` event, checking the
        :attr:`.Session.is_active` flag.

        :param session: The target :class:`.Session`.

        """