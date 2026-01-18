from __future__ import annotations
from copy import deepcopy
from typing import (
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions
from bson.dbref import DBRef
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import _csot, common
from pymongo.aggregation import _DatabaseAggregationCommand
from pymongo.change_stream import DatabaseChangeStream
from pymongo.collection import Collection
from pymongo.command_cursor import CommandCursor
from pymongo.common import _ecoc_coll_name, _esc_coll_name
from pymongo.errors import CollectionInvalid, InvalidName, InvalidOperation
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.typings import _CollationIn, _DocumentType, _DocumentTypeArg, _Pipeline
def list_collection_names(self, session: Optional[ClientSession]=None, filter: Optional[Mapping[str, Any]]=None, comment: Optional[Any]=None, **kwargs: Any) -> list[str]:
    """Get a list of all the collection names in this database.

        For example, to list all non-system collections::

            filter = {"name": {"$regex": r"^(?!system\\.)"}}
            db.list_collection_names(filter=filter)

        :Parameters:
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`.
          - `filter` (optional):  A query document to filter the list of
            collections returned from the listCollections command.
          - `comment` (optional): A user-provided comment to attach to this
            command.
          - `**kwargs` (optional): Optional parameters of the
            `listCollections command
            <https://mongodb.com/docs/manual/reference/command/listCollections/>`_
            can be passed as keyword arguments to this method. The supported
            options differ by server version.


        .. versionchanged:: 3.8
           Added the ``filter`` and ``**kwargs`` parameters.

        .. versionadded:: 3.6
        """
    if comment is not None:
        kwargs['comment'] = comment
    if filter is None:
        kwargs['nameOnly'] = True
    else:
        common.validate_is_mapping('filter', filter)
        kwargs['filter'] = filter
        if not filter or (len(filter) == 1 and 'name' in filter):
            kwargs['nameOnly'] = True
    return [result['name'] for result in self.list_collections(session=session, **kwargs)]