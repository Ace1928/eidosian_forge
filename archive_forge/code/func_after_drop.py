from __future__ import annotations
from typing import Any
from typing import TYPE_CHECKING
from .base import SchemaEventTarget
from .. import event
def after_drop(self, target: SchemaEventTarget, connection: Connection, **kw: Any) -> None:
    """Called after DROP statements are emitted.

        :param target: the :class:`.SchemaObject`, such as a
         :class:`_schema.MetaData` or :class:`_schema.Table`
         but also including all create/drop objects such as
         :class:`.Index`, :class:`.Sequence`, etc.,
         object which is the target of the event.

         .. versionadded:: 2.0 Support for all :class:`.SchemaItem` objects
            was added.

        :param connection: the :class:`_engine.Connection` where the
         DROP statement or statements have been emitted.
        :param \\**kw: additional keyword arguments relevant
         to the event.  The contents of this dictionary
         may vary across releases, and include the
         list of tables being generated for a metadata-level
         event, the checkfirst flag, and other
         elements used by internal events.

        :func:`.event.listen` also accepts the ``propagate=True``
        modifier for this event; when True, the listener function will
        be established for any copies made of the target object,
        i.e. those copies that are generated when
        :meth:`_schema.Table.to_metadata` is used.

        """