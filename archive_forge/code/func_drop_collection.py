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
@_csot.apply
def drop_collection(self, name_or_collection: Union[str, Collection[_DocumentTypeArg]], session: Optional[ClientSession]=None, comment: Optional[Any]=None, encrypted_fields: Optional[Mapping[str, Any]]=None) -> dict[str, Any]:
    """Drop a collection.

        :Parameters:
          - `name_or_collection`: the name of a collection to drop or the
            collection object itself
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`.
          - `comment` (optional): A user-provided comment to attach to this
            command.
          - `encrypted_fields`: **(BETA)** Document that describes the encrypted fields for
            Queryable Encryption. For example::

                {
                  "escCollection": "enxcol_.encryptedCollection.esc",
                  "ecocCollection": "enxcol_.encryptedCollection.ecoc",
                  "fields": [
                      {
                          "path": "firstName",
                          "keyId": Binary.from_uuid(UUID('00000000-0000-0000-0000-000000000000')),
                          "bsonType": "string",
                          "queries": {"queryType": "equality"}
                      },
                      {
                          "path": "ssn",
                          "keyId": Binary.from_uuid(UUID('04104104-1041-0410-4104-104104104104')),
                          "bsonType": "string"
                      }
                  ]

                }


        .. note:: The :attr:`~pymongo.database.Database.write_concern` of
           this database is automatically applied to this operation.

        .. versionchanged:: 4.2
           Added ``encrypted_fields`` parameter.

        .. versionchanged:: 4.1
           Added ``comment`` parameter.

        .. versionchanged:: 3.6
           Added ``session`` parameter.

        .. versionchanged:: 3.4
           Apply this database's write concern automatically to this operation
           when connected to MongoDB >= 3.4.

        """
    name = name_or_collection
    if isinstance(name, Collection):
        name = name.name
    if not isinstance(name, str):
        raise TypeError('name_or_collection must be an instance of str')
    encrypted_fields = self._get_encrypted_fields({'encryptedFields': encrypted_fields}, name, True)
    if encrypted_fields:
        common.validate_is_mapping('encrypted_fields', encrypted_fields)
        self._drop_helper(_esc_coll_name(encrypted_fields, name), session=session, comment=comment)
        self._drop_helper(_ecoc_coll_name(encrypted_fields, name), session=session, comment=comment)
    return self._drop_helper(name, session, comment)