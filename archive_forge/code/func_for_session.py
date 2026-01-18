import collections.abc as collections_abc
import logging
from .. import exc as sa_exc
from .. import util
from ..orm import exc as orm_exc
from ..orm.query import Query
from ..orm.session import Session
from ..sql import func
from ..sql import literal_column
from ..sql import util as sql_util
def for_session(self, session):
    """Return a :class:`_baked.Result` object for this
        :class:`.BakedQuery`.

        This is equivalent to calling the :class:`.BakedQuery` as a
        Python callable, e.g. ``result = my_baked_query(session)``.

        """
    return Result(self, session)