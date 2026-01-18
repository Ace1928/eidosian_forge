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
def do_orm_execute(self, orm_execute_state: ORMExecuteState) -> None:
    """Intercept statement executions that occur on behalf of an
        ORM :class:`.Session` object.

        This event is invoked for all top-level SQL statements invoked from the
        :meth:`_orm.Session.execute` method, as well as related methods such as
        :meth:`_orm.Session.scalars` and :meth:`_orm.Session.scalar`. As of
        SQLAlchemy 1.4, all ORM queries that run through the
        :meth:`_orm.Session.execute` method as well as related methods
        :meth:`_orm.Session.scalars`, :meth:`_orm.Session.scalar` etc.
        will participate in this event.
        This event hook does **not** apply to the queries that are
        emitted internally within the ORM flush process, i.e. the
        process described at :ref:`session_flushing`.

        .. note::  The :meth:`_orm.SessionEvents.do_orm_execute` event hook
           is triggered **for ORM statement executions only**, meaning those
           invoked via the :meth:`_orm.Session.execute` and similar methods on
           the :class:`_orm.Session` object. It does **not** trigger for
           statements that are invoked by SQLAlchemy Core only, i.e. statements
           invoked directly using :meth:`_engine.Connection.execute` or
           otherwise originating from an :class:`_engine.Engine` object without
           any :class:`_orm.Session` involved. To intercept **all** SQL
           executions regardless of whether the Core or ORM APIs are in use,
           see the event hooks at :class:`.ConnectionEvents`, such as
           :meth:`.ConnectionEvents.before_execute` and
           :meth:`.ConnectionEvents.before_cursor_execute`.

           Also, this event hook does **not** apply to queries that are
           emitted internally within the ORM flush process,
           i.e. the process described at :ref:`session_flushing`; to
           intercept steps within the flush process, see the event
           hooks described at :ref:`session_persistence_events` as
           well as :ref:`session_persistence_mapper`.

        This event is a ``do_`` event, meaning it has the capability to replace
        the operation that the :meth:`_orm.Session.execute` method normally
        performs.  The intended use for this includes sharding and
        result-caching schemes which may seek to invoke the same statement
        across  multiple database connections, returning a result that is
        merged from each of them, or which don't invoke the statement at all,
        instead returning data from a cache.

        The hook intends to replace the use of the
        ``Query._execute_and_instances`` method that could be subclassed prior
        to SQLAlchemy 1.4.

        :param orm_execute_state: an instance of :class:`.ORMExecuteState`
         which contains all information about the current execution, as well
         as helper functions used to derive other commonly required
         information.   See that object for details.

        .. seealso::

            :ref:`session_execute_events` - top level documentation on how
            to use :meth:`_orm.SessionEvents.do_orm_execute`

            :class:`.ORMExecuteState` - the object passed to the
            :meth:`_orm.SessionEvents.do_orm_execute` event which contains
            all information about the statement to be invoked.  It also
            provides an interface to extend the current statement, options,
            and parameters as well as an option that allows programmatic
            invocation of the statement at any point.

            :ref:`examples_session_orm_events` - includes examples of using
            :meth:`_orm.SessionEvents.do_orm_execute`

            :ref:`examples_caching` - an example of how to integrate
            Dogpile caching with the ORM :class:`_orm.Session` making use
            of the :meth:`_orm.SessionEvents.do_orm_execute` event hook.

            :ref:`examples_sharding` - the Horizontal Sharding example /
            extension relies upon the
            :meth:`_orm.SessionEvents.do_orm_execute` event hook to invoke a
            SQL statement on multiple backends and return a merged result.


        .. versionadded:: 1.4

        """