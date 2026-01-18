from __future__ import annotations
from lazyops.libs.dbinit.base import Engine
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.entities.role import Role
@classmethod
def remove_all(cls, engine: Engine | None=None) -> None:
    """
        Attempts to revoke all declared privileges then drop all declared entities. Main way to do so.
        :param engine: A :class:`sqlalchemy.Engine` that defines the connection to a Postgres instance/cluster.
        """
    cls._handle_engine(engine)
    cls.revoke_all()
    cls.drop_all()