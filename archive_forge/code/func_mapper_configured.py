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
def mapper_configured(self, mapper: Mapper[_O], class_: Type[_O]) -> None:
    """Called when a specific mapper has completed its own configuration
        within the scope of the :func:`.configure_mappers` call.

        The :meth:`.MapperEvents.mapper_configured` event is invoked
        for each mapper that is encountered when the
        :func:`_orm.configure_mappers` function proceeds through the current
        list of not-yet-configured mappers.
        :func:`_orm.configure_mappers` is typically invoked
        automatically as mappings are first used, as well as each time
        new mappers have been made available and new mapper use is
        detected.

        When the event is called, the mapper should be in its final
        state, but **not including backrefs** that may be invoked from
        other mappers; they might still be pending within the
        configuration operation.    Bidirectional relationships that
        are instead configured via the
        :paramref:`.orm.relationship.back_populates` argument
        *will* be fully available, since this style of relationship does not
        rely upon other possibly-not-configured mappers to know that they
        exist.

        For an event that is guaranteed to have **all** mappers ready
        to go including backrefs that are defined only on other
        mappings, use the :meth:`.MapperEvents.after_configured`
        event; this event invokes only after all known mappings have been
        fully configured.

        The :meth:`.MapperEvents.mapper_configured` event, unlike
        :meth:`.MapperEvents.before_configured` or
        :meth:`.MapperEvents.after_configured`,
        is called for each mapper/class individually, and the mapper is
        passed to the event itself.  It also is called exactly once for
        a particular mapper.  The event is therefore useful for
        configurational steps that benefit from being invoked just once
        on a specific mapper basis, which don't require that "backref"
        configurations are necessarily ready yet.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param class\\_: the mapped class.

        .. seealso::

            :meth:`.MapperEvents.before_configured`

            :meth:`.MapperEvents.after_configured`

            :meth:`.MapperEvents.before_mapper_configured`

        """