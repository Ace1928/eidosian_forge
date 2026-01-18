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
def refresh_flush(self, target: _O, flush_context: UOWTransaction, attrs: Optional[Iterable[str]]) -> None:
    """Receive an object instance after one or more attributes that
        contain a column-level default or onupdate handler have been refreshed
        during persistence of the object's state.

        This event is the same as :meth:`.InstanceEvents.refresh` except
        it is invoked within the unit of work flush process, and includes
        only non-primary-key columns that have column level default or
        onupdate handlers, including Python callables as well as server side
        defaults and triggers which may be fetched via the RETURNING clause.

        .. note::

            While the :meth:`.InstanceEvents.refresh_flush` event is triggered
            for an object that was INSERTed as well as for an object that was
            UPDATEd, the event is geared primarily  towards the UPDATE process;
            it is mostly an internal artifact that INSERT actions can also
            trigger this event, and note that **primary key columns for an
            INSERTed row are explicitly omitted** from this event.  In order to
            intercept the newly INSERTed state of an object, the
            :meth:`.SessionEvents.pending_to_persistent` and
            :meth:`.MapperEvents.after_insert` are better choices.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param flush_context: Internal :class:`.UOWTransaction` object
         which handles the details of the flush.
        :param attrs: sequence of attribute names which
         were populated.

        .. seealso::

            :ref:`mapped_class_load_events`

            :ref:`orm_server_defaults`

            :ref:`metadata_defaults_toplevel`

        """