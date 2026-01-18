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
def insert_one(self, document: Union[_DocumentType, RawBSONDocument], bypass_document_validation: bool=False, session: Optional[ClientSession]=None, comment: Optional[Any]=None) -> InsertOneResult:
    """Insert a single document.

          >>> db.test.count_documents({'x': 1})
          0
          >>> result = db.test.insert_one({'x': 1})
          >>> result.inserted_id
          ObjectId('54f112defba522406c9cc208')
          >>> db.test.find_one({'x': 1})
          {'x': 1, '_id': ObjectId('54f112defba522406c9cc208')}

        :Parameters:
          - `document`: The document to insert. Must be a mutable mapping
            type. If the document does not have an _id field one will be
            added automatically.
          - `bypass_document_validation`: (optional) If ``True``, allows the
            write to opt-out of document level validation. Default is
            ``False``.
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`.
          - `comment` (optional): A user-provided comment to attach to this
            command.

        :Returns:
          - An instance of :class:`~pymongo.results.InsertOneResult`.

        .. seealso:: :ref:`writes-and-ids`

        .. note:: `bypass_document_validation` requires server version
          **>= 3.2**

        .. versionchanged:: 4.1
           Added ``comment`` parameter.

        .. versionchanged:: 3.6
           Added ``session`` parameter.

        .. versionchanged:: 3.2
          Added bypass_document_validation support

        .. versionadded:: 3.0
        """
    common.validate_is_document_type('document', document)
    if not (isinstance(document, RawBSONDocument) or '_id' in document):
        document['_id'] = ObjectId()
    write_concern = self._write_concern_for(session)
    return InsertOneResult(self._insert_one(document, ordered=True, write_concern=write_concern, op_id=None, bypass_doc_val=bypass_document_validation, session=session, comment=comment), write_concern.acknowledged)