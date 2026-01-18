from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
@property
def inserted_ids(self) -> list[Any]:
    """A list of _ids of the inserted documents, in the order provided.

        .. note:: If ``False`` is passed for the `ordered` parameter to
          :meth:`~pymongo.collection.Collection.insert_many` the server
          may have inserted the documents in a different order than what
          is presented here.
        """
    return self.__inserted_ids