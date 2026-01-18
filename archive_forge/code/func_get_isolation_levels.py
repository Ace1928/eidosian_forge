from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
def get_isolation_levels(self, config):
    """Return a structure of supported isolation levels for the current
        testing dialect.

        The structure indicates to the testing suite what the expected
        "default" isolation should be, as well as the other values that
        are accepted.  The dictionary has two keys, "default" and "supported".
        The "supported" key refers to a list of all supported levels and
        it should include AUTOCOMMIT if the dialect supports it.

        If the :meth:`.DefaultRequirements.isolation_level` requirement is
        not open, then this method has no return value.

        E.g.::

            >>> testing.requirements.get_isolation_levels()
            {
                "default": "READ_COMMITTED",
                "supported": [
                    "SERIALIZABLE", "READ UNCOMMITTED",
                    "READ COMMITTED", "REPEATABLE READ",
                    "AUTOCOMMIT"
                ]
            }
        """
    with config.db.connect() as conn:
        try:
            supported = conn.dialect.get_isolation_level_values(conn.connection.dbapi_connection)
        except NotImplementedError:
            return None
        else:
            return {'default': conn.dialect.default_isolation_level, 'supported': supported}