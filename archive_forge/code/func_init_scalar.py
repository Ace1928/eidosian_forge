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
def init_scalar(self, target: _O, value: _T, dict_: Dict[Any, Any]) -> None:
    """Receive a scalar "init" event.

        This event is invoked when an uninitialized, unpersisted scalar
        attribute is accessed, e.g. read::


            x = my_object.some_attribute

        The ORM's default behavior when this occurs for an un-initialized
        attribute is to return the value ``None``; note this differs from
        Python's usual behavior of raising ``AttributeError``.    The
        event here can be used to customize what value is actually returned,
        with the assumption that the event listener would be mirroring
        a default generator that is configured on the Core
        :class:`_schema.Column`
        object as well.

        Since a default generator on a :class:`_schema.Column`
        might also produce
        a changing value such as a timestamp, the
        :meth:`.AttributeEvents.init_scalar`
        event handler can also be used to **set** the newly returned value, so
        that a Core-level default generation function effectively fires off
        only once, but at the moment the attribute is accessed on the
        non-persisted object.   Normally, no change to the object's state
        is made when an uninitialized attribute is accessed (much older
        SQLAlchemy versions did in fact change the object's state).

        If a default generator on a column returned a particular constant,
        a handler might be used as follows::

            SOME_CONSTANT = 3.1415926

            class MyClass(Base):
                # ...

                some_attribute = Column(Numeric, default=SOME_CONSTANT)

            @event.listens_for(
                MyClass.some_attribute, "init_scalar",
                retval=True, propagate=True)
            def _init_some_attribute(target, dict_, value):
                dict_['some_attribute'] = SOME_CONSTANT
                return SOME_CONSTANT

        Above, we initialize the attribute ``MyClass.some_attribute`` to the
        value of ``SOME_CONSTANT``.   The above code includes the following
        features:

        * By setting the value ``SOME_CONSTANT`` in the given ``dict_``,
          we indicate that this value is to be persisted to the database.
          This supersedes the use of ``SOME_CONSTANT`` in the default generator
          for the :class:`_schema.Column`.  The ``active_column_defaults.py``
          example given at :ref:`examples_instrumentation` illustrates using
          the same approach for a changing default, e.g. a timestamp
          generator.    In this particular example, it is not strictly
          necessary to do this since ``SOME_CONSTANT`` would be part of the
          INSERT statement in either case.

        * By establishing the ``retval=True`` flag, the value we return
          from the function will be returned by the attribute getter.
          Without this flag, the event is assumed to be a passive observer
          and the return value of our function is ignored.

        * The ``propagate=True`` flag is significant if the mapped class
          includes inheriting subclasses, which would also make use of this
          event listener.  Without this flag, an inheriting subclass will
          not use our event handler.

        In the above example, the attribute set event
        :meth:`.AttributeEvents.set` as well as the related validation feature
        provided by :obj:`_orm.validates` is **not** invoked when we apply our
        value to the given ``dict_``.  To have these events to invoke in
        response to our newly generated value, apply the value to the given
        object as a normal attribute set operation::

            SOME_CONSTANT = 3.1415926

            @event.listens_for(
                MyClass.some_attribute, "init_scalar",
                retval=True, propagate=True)
            def _init_some_attribute(target, dict_, value):
                # will also fire off attribute set events
                target.some_attribute = SOME_CONSTANT
                return SOME_CONSTANT

        When multiple listeners are set up, the generation of the value
        is "chained" from one listener to the next by passing the value
        returned by the previous listener that specifies ``retval=True``
        as the ``value`` argument of the next listener.

        :param target: the object instance receiving the event.
         If the listener is registered with ``raw=True``, this will
         be the :class:`.InstanceState` object.
        :param value: the value that is to be returned before this event
         listener were invoked.  This value begins as the value ``None``,
         however will be the return value of the previous event handler
         function if multiple listeners are present.
        :param dict\\_: the attribute dictionary of this mapped object.
         This is normally the ``__dict__`` of the object, but in all cases
         represents the destination that the attribute system uses to get
         at the actual value of this attribute.  Placing the value in this
         dictionary has the effect that the value will be used in the
         INSERT statement generated by the unit of work.


        .. seealso::

            :meth:`.AttributeEvents.init_collection` - collection version
            of this event

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

            :ref:`examples_instrumentation` - see the
            ``active_column_defaults.py`` example.

        """