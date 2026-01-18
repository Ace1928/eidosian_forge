from __future__ import annotations
import weakref
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.asyncio import AsyncSession, async_object_session
def receive_after_commit(session):
    """Reinstate objects into their previous sessions."""
    objects = filter(lambda o: hasattr(o, '_prev_session'), session.identity_map.values())
    for object_ in objects:
        prev_session = object_._prev_session()
        if prev_session:
            session.expunge(object_)
            prev_session.add(object_)
            delattr(object_, '_prev_session')