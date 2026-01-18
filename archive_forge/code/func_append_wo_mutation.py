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
def append_wo_mutation(self, target: _O, value: _T, initiator: Event, *, key: EventConstants=NO_KEY) -> None:
    """Receive a collection append event where the collection was not
        actually mutated.

        This event differs from :meth:`_orm.AttributeEvents.append` in that
        it is fired off for de-duplicating collections such as sets and
        dictionaries, when the object already exists in the target collection.
        The event does not have a return value and the identity of the
        given object cannot be changed.

        The event is used for cascading objects into a :class:`_orm.Session`
        when the collection has already been mutated via a backref event.

        :param target: the object instance receiving the event.
          If the listener is registered with ``raw=True``, this will
          be the :class:`.InstanceState` object.
        :param value: the value that would be appended if the object did not
          already exist in the collection.
        :param initiator: An instance of :class:`.attributes.Event`
          representing the initiation of the event.  May be modified
          from its original value by backref handlers in order to control
          chained event propagation, as well as be inspected for information
          about the source of the event.
        :param key: When the event is established using the
         :paramref:`.AttributeEvents.include_key` parameter set to
         True, this will be the key used in the operation, such as
         ``collection[some_key_or_index] = value``.
         The parameter is not passed
         to the event at all if the the
         :paramref:`.AttributeEvents.include_key`
         was not used to set up the event; this is to allow backwards
         compatibility with existing event handlers that don't include the
         ``key`` parameter.

         .. versionadded:: 2.0

        :return: No return value is defined for this event.

        .. versionadded:: 1.4.15

        """