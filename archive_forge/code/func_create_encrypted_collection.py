from __future__ import annotations
import contextlib
import enum
import socket
import weakref
from copy import deepcopy
from typing import (
from bson import _dict_to_bson, decode, encode
from bson.binary import STANDARD, UUID_SUBTYPE, Binary
from bson.codec_options import CodecOptions
from bson.errors import BSONError
from bson.raw_bson import DEFAULT_RAW_BSON_OPTIONS, RawBSONDocument, _inflate_bson
from bson.son import SON
from pymongo import _csot
from pymongo.collection import Collection
from pymongo.common import CONNECT_TIMEOUT
from pymongo.cursor import Cursor
from pymongo.daemon import _spawn_daemon
from pymongo.database import Database
from pymongo.encryption_options import AutoEncryptionOpts, RangeOpts
from pymongo.errors import (
from pymongo.mongo_client import MongoClient
from pymongo.network import BLOCKING_IO_ERRORS
from pymongo.operations import UpdateOne
from pymongo.pool import PoolOptions, _configured_socket, _raise_connection_failure
from pymongo.read_concern import ReadConcern
from pymongo.results import BulkWriteResult, DeleteResult
from pymongo.ssl_support import get_ssl_context
from pymongo.typings import _DocumentType, _DocumentTypeArg
from pymongo.uri_parser import parse_host
from pymongo.write_concern import WriteConcern
def create_encrypted_collection(self, database: Database[_DocumentTypeArg], name: str, encrypted_fields: Mapping[str, Any], kms_provider: Optional[str]=None, master_key: Optional[Mapping[str, Any]]=None, **kwargs: Any) -> tuple[Collection[_DocumentTypeArg], Mapping[str, Any]]:
    """Create a collection with encryptedFields.

        .. warning::
            This function does not update the encryptedFieldsMap in the client's
            AutoEncryptionOpts, thus the user must create a new client after calling this function with
            the encryptedFields returned.

        Normally collection creation is automatic. This method should
        only be used to specify options on
        creation. :class:`~pymongo.errors.EncryptionError` will be
        raised if the collection already exists.

        :Parameters:
          - `name`: the name of the collection to create
          - `encrypted_fields` (dict): Document that describes the encrypted fields for
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

            The "keyId" may be set to ``None`` to auto-generate the data keys.
          - `kms_provider` (optional): the KMS provider to be used
          - `master_key` (optional): Identifies a KMS-specific key used to encrypt the
            new data key. If the kmsProvider is "local" the `master_key` is
            not applicable and may be omitted.
          - `**kwargs` (optional): additional keyword arguments are the same as "create_collection".

        All optional `create collection command`_ parameters should be passed
        as keyword arguments to this method.
        See the documentation for :meth:`~pymongo.database.Database.create_collection` for all valid options.

        :Raises:
          - :class:`~pymongo.errors.EncryptedCollectionError`: When either data-key creation or creating the collection fails.

        .. versionadded:: 4.4

        .. _create collection command:
            https://mongodb.com/docs/manual/reference/command/create

        """
    encrypted_fields = deepcopy(encrypted_fields)
    for i, field in enumerate(encrypted_fields['fields']):
        if isinstance(field, dict) and field.get('keyId') is None:
            try:
                encrypted_fields['fields'][i]['keyId'] = self.create_data_key(kms_provider=kms_provider, master_key=master_key)
            except EncryptionError as exc:
                raise EncryptedCollectionError(exc, encrypted_fields) from exc
    kwargs['encryptedFields'] = encrypted_fields
    kwargs['check_exists'] = False
    try:
        return (database.create_collection(name=name, **kwargs), encrypted_fields)
    except Exception as exc:
        raise EncryptedCollectionError(exc, encrypted_fields) from exc