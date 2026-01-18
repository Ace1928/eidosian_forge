from __future__ import annotations
from lazyops.libs.dbinit.base import Engine
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.entities.role import Role
@classmethod
def grant_all(cls, engine: Engine | None=None) -> None:
    """
        Attempts to grant all declared privileges. Requires entities to exist, typically run via `run_all` or after
        `create_all`.
        :param engine: A :class:`sqlalchemy.Engine` that defines the connection to a Postgres instance/cluster.
        """
    cls._handle_engine(engine)
    for entity in Entity.entities:
        if isinstance(entity, Role):
            entity._safe_grant()