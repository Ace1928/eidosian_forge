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
@event._legacy_signature('0.9', ['session', 'query', 'query_context', 'result'], lambda delete_context: (delete_context.session, delete_context.query, None, delete_context.result))
def after_bulk_delete(self, delete_context: _O) -> None:
    """Event for after the legacy :meth:`_orm.Query.delete` method
        has been called.

        .. legacy:: The :meth:`_orm.SessionEvents.after_bulk_delete` method
           is a legacy event hook as of SQLAlchemy 2.0.   The event
           **does not participate** in :term:`2.0 style` invocations
           using :func:`_dml.delete` documented at
           :ref:`orm_queryguide_update_delete_where`.  For 2.0 style use,
           the :meth:`_orm.SessionEvents.do_orm_execute` hook will intercept
           these calls.

        :param delete_context: a "delete context" object which contains
         details about the update, including these attributes:

            * ``session`` - the :class:`.Session` involved
            * ``query`` -the :class:`_query.Query`
              object that this update operation
              was called upon.
            * ``result`` the :class:`_engine.CursorResult`
              returned as a result of the
              bulk DELETE operation.

        .. versionchanged:: 1.4 the update_context no longer has a
           ``QueryContext`` object associated with it.

        .. seealso::

            :meth:`.QueryEvents.before_compile_delete`

            :meth:`.SessionEvents.after_bulk_update`

        """