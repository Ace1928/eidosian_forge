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
def count_documents(self, filter: Mapping[str, Any], session: Optional[ClientSession]=None, comment: Optional[Any]=None, **kwargs: Any) -> int:
    """Count the number of documents in this collection.

        .. note:: For a fast count of the total documents in a collection see
           :meth:`estimated_document_count`.

        The :meth:`count_documents` method is supported in a transaction.

        All optional parameters should be passed as keyword arguments
        to this method. Valid options include:

          - `skip` (int): The number of matching documents to skip before
            returning results.
          - `limit` (int): The maximum number of documents to count. Must be
            a positive integer. If not provided, no limit is imposed.
          - `maxTimeMS` (int): The maximum amount of time to allow this
            operation to run, in milliseconds.
          - `collation` (optional): An instance of
            :class:`~pymongo.collation.Collation`.
          - `hint` (string or list of tuples): The index to use. Specify either
            the index name as a string or the index specification as a list of
            tuples (e.g. [('a', pymongo.ASCENDING), ('b', pymongo.ASCENDING)]).

        The :meth:`count_documents` method obeys the :attr:`read_preference` of
        this :class:`Collection`.

        .. note:: When migrating from :meth:`count` to :meth:`count_documents`
           the following query operators must be replaced:

           +-------------+-------------------------------------+
           | Operator    | Replacement                         |
           +=============+=====================================+
           | $where      | `$expr`_                            |
           +-------------+-------------------------------------+
           | $near       | `$geoWithin`_ with `$center`_       |
           +-------------+-------------------------------------+
           | $nearSphere | `$geoWithin`_ with `$centerSphere`_ |
           +-------------+-------------------------------------+

        :Parameters:
          - `filter` (required): A query document that selects which documents
            to count in the collection. Can be an empty document to count all
            documents.
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`.
          - `comment` (optional): A user-provided comment to attach to this
            command.
          - `**kwargs` (optional): See list of options above.


        .. versionadded:: 3.7

        .. _$expr: https://mongodb.com/docs/manual/reference/operator/query/expr/
        .. _$geoWithin: https://mongodb.com/docs/manual/reference/operator/query/geoWithin/
        .. _$center: https://mongodb.com/docs/manual/reference/operator/query/center/
        .. _$centerSphere: https://mongodb.com/docs/manual/reference/operator/query/centerSphere/
        """
    pipeline = [{'$match': filter}]
    if 'skip' in kwargs:
        pipeline.append({'$skip': kwargs.pop('skip')})
    if 'limit' in kwargs:
        pipeline.append({'$limit': kwargs.pop('limit')})
    if comment is not None:
        kwargs['comment'] = comment
    pipeline.append({'$group': {'_id': 1, 'n': {'$sum': 1}}})
    cmd = SON([('aggregate', self.__name), ('pipeline', pipeline), ('cursor', {})])
    if 'hint' in kwargs and (not isinstance(kwargs['hint'], str)):
        kwargs['hint'] = helpers._index_document(kwargs['hint'])
    collation = validate_collation_or_none(kwargs.pop('collation', None))
    cmd.update(kwargs)

    def _cmd(session: Optional[ClientSession], _server: Server, conn: Connection, read_preference: Optional[_ServerMode]) -> int:
        result = self._aggregate_one_result(conn, read_preference, cmd, collation, session)
        if not result:
            return 0
        return result['n']
    return self._retryable_non_cursor_read(_cmd, session)