from __future__ import annotations
from collections import abc
from typing import (
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions
from bson.objectid import ObjectId
from bson.raw_bson import RawBSONDocument
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import ASCENDING, _csot, common, helpers, message
from pymongo.aggregation import (
from pymongo.bulk import _Bulk
from pymongo.change_stream import CollectionChangeStream
from pymongo.collation import validate_collation_or_none
from pymongo.command_cursor import CommandCursor, RawBatchCommandCursor
from pymongo.common import _ecoc_coll_name, _esc_coll_name
from pymongo.cursor import Cursor, RawBatchCursor
from pymongo.errors import (
from pymongo.helpers import _check_write_command_response
from pymongo.message import _UNICODE_REPLACE_CODEC_OPTIONS
from pymongo.operations import (
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.results import (
from pymongo.typings import _CollationIn, _DocumentType, _DocumentTypeArg, _Pipeline
from pymongo.write_concern import WriteConcern
def list_search_indexes(self, name: Optional[str]=None, session: Optional[ClientSession]=None, comment: Optional[Any]=None, **kwargs: Any) -> CommandCursor[Mapping[str, Any]]:
    """Return a cursor over search indexes for the current collection.

        :Parameters:
          - `name` (optional): If given, the name of the index to search
            for.  Only indexes with matching index names will be returned.
            If not given, all search indexes for the current collection
            will be returned.
          - `session` (optional): a :class:`~pymongo.client_session.ClientSession`.
          - `comment` (optional): A user-provided comment to attach to this
            command.

        :Returns:
          A :class:`~pymongo.command_cursor.CommandCursor` over the result
          set.

        .. note:: requires a MongoDB server version 7.0+ Atlas cluster.

        .. versionadded:: 4.5
        """
    if name is None:
        pipeline: _Pipeline = [{'$listSearchIndexes': {}}]
    else:
        pipeline = [{'$listSearchIndexes': {'name': name}}]
    coll = self.with_options(codec_options=DEFAULT_CODEC_OPTIONS, read_preference=ReadPreference.PRIMARY)
    cmd = _CollectionAggregationCommand(coll, CommandCursor, pipeline, kwargs, explicit_session=session is not None, user_fields={'cursor': {'firstBatch': 1}})
    return self.__database.client._retryable_read(cmd.get_cursor, cmd.get_read_preference(session), session, retryable=not cmd._performs_write)