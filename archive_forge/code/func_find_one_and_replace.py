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
def find_one_and_replace(self, filter: Mapping[str, Any], replacement: Mapping[str, Any], projection: Optional[Union[Mapping[str, Any], Iterable[str]]]=None, sort: Optional[_IndexList]=None, upsert: bool=False, return_document: bool=ReturnDocument.BEFORE, hint: Optional[_IndexKeyHint]=None, session: Optional[ClientSession]=None, let: Optional[Mapping[str, Any]]=None, comment: Optional[Any]=None, **kwargs: Any) -> _DocumentType:
    """Finds a single document and replaces it, returning either the
        original or the replaced document.

        The :meth:`find_one_and_replace` method differs from
        :meth:`find_one_and_update` by replacing the document matched by
        *filter*, rather than modifying the existing document.

          >>> for doc in db.test.find({}):
          ...     print(doc)
          ...
          {'x': 1, '_id': 0}
          {'x': 1, '_id': 1}
          {'x': 1, '_id': 2}
          >>> db.test.find_one_and_replace({'x': 1}, {'y': 1})
          {'x': 1, '_id': 0}
          >>> for doc in db.test.find({}):
          ...     print(doc)
          ...
          {'y': 1, '_id': 0}
          {'x': 1, '_id': 1}
          {'x': 1, '_id': 2}

        :Parameters:
          - `filter`: A query that matches the document to replace.
          - `replacement`: The replacement document.
          - `projection` (optional): A list of field names that should be
            returned in the result document or a mapping specifying the fields
            to include or exclude. If `projection` is a list "_id" will
            always be returned. Use a mapping to exclude fields from
            the result (e.g. projection={'_id': False}).
          - `sort` (optional): a list of (key, direction) pairs
            specifying the sort order for the query. If multiple documents
            match the query, they are sorted and the first is replaced.
          - `upsert` (optional): When ``True``, inserts a new document if no
            document matches the query. Defaults to ``False``.
          - `return_document`: If
            :attr:`ReturnDocument.BEFORE` (the default),
            returns the original document before it was replaced, or ``None``
            if no document matches. If
            :attr:`ReturnDocument.AFTER`, returns the replaced
            or inserted document.
          - `hint` (optional): An index to use to support the query
            predicate specified either by its string name, or in the same
            format as passed to
            :meth:`~pymongo.collection.Collection.create_index` (e.g.
            ``[('field', ASCENDING)]``). This option is only supported on
            MongoDB 4.4 and above.
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`.
          - `let` (optional): Map of parameter names and values. Values must be
            constant or closed expressions that do not reference document
            fields. Parameters can then be accessed as variables in an
            aggregate expression context (e.g. "$$var").
          - `comment` (optional): A user-provided comment to attach to this
            command.
          - `**kwargs` (optional): additional command arguments can be passed
            as keyword arguments (for example maxTimeMS can be used with
            recent server versions).

        .. versionchanged:: 4.1
           Added ``let`` parameter.
        .. versionchanged:: 3.11
           Added the ``hint`` option.
        .. versionchanged:: 3.6
           Added ``session`` parameter.
        .. versionchanged:: 3.4
           Added the ``collation`` option.
        .. versionchanged:: 3.2
           Respects write concern.

        .. warning:: Starting in PyMongo 3.2, this command uses the
           :class:`~pymongo.write_concern.WriteConcern` of this
           :class:`~pymongo.collection.Collection` when connected to MongoDB >=
           3.2. Note that using an elevated write concern with this command may
           be slower compared to using the default write concern.

        .. versionadded:: 3.0
        """
    common.validate_ok_for_replace(replacement)
    kwargs['update'] = replacement
    if comment is not None:
        kwargs['comment'] = comment
    return self.__find_and_modify(filter, projection, sort, upsert, return_document, let=let, hint=hint, session=session, **kwargs)