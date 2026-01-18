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
@classmethod
def bakery(cls, size=200, _size_alert=None):
    """Construct a new bakery.

        :return: an instance of :class:`.Bakery`

        """
    return Bakery(cls, util.LRUCache(size, size_alert=_size_alert))