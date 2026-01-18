from __future__ import annotations
from typing import Any
from typing import TYPE_CHECKING
from .base import SchemaEventTarget
from .. import event
def before_parent_attach(self, target: SchemaEventTarget, parent: SchemaItem) -> None:
    """Called before a :class:`.SchemaItem` is associated with
        a parent :class:`.SchemaItem`.

        :param target: the target object
        :param parent: the parent to which the target is being attached.

        :func:`.event.listen` also accepts the ``propagate=True``
        modifier for this event; when True, the listener function will
        be established for any copies made of the target object,
        i.e. those copies that are generated when
        :meth:`_schema.Table.to_metadata` is used.

        """