from __future__ import annotations
from lazyops.libs.dbinit.base import Engine
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.entities.role import Role
@classmethod
def revoke_all(cls, engine: Engine | None=None) -> None:
    """
        Attempts to revoke all declared privileges. Requires entities to exist, typically run via `remove_all` or before
        `drop_all`.
        :param engine: A :class:`sqlalchemy.Engine` that defines the connection to a Postgres instance/cluster.
        """
    cls._handle_engine(engine)
    for entity in reversed(Entity.entities):
        if isinstance(entity, Role):
            entity._safe_revoke()