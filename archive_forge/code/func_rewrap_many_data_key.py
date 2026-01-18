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
def rewrap_many_data_key(self, filter: Mapping[str, Any], provider: Optional[str]=None, master_key: Optional[Mapping[str, Any]]=None) -> RewrapManyDataKeyResult:
    """Decrypts and encrypts all matching data keys in the key vault with a possibly new `master_key` value.

        :Parameters:
          - `filter`: A document used to filter the data keys.
          - `provider`: The new KMS provider to use to encrypt the data keys,
            or ``None`` to use the current KMS provider(s).
          - ``master_key``: The master key fields corresponding to the new KMS
            provider when ``provider`` is not ``None``.

        :Returns:
          A :class:`RewrapManyDataKeyResult`.

        This method allows you to re-encrypt all of your data-keys with a new CMK, or master key.
        Note that this does *not* require re-encrypting any of the data in your encrypted collections,
        but rather refreshes the key that protects the keys that encrypt the data:

        .. code-block:: python

           client_encryption.rewrap_many_data_key(
               filter={"keyAltNames": "optional filter for which keys you want to update"},
               master_key={
                   "provider": "azure",  # replace with your cloud provider
                   "master_key": {
                       # put the rest of your master_key options here
                       "key": "<your new key>"
                   },
               },
           )

        .. versionadded:: 4.2
        """
    if master_key is not None and provider is None:
        raise ConfigurationError('A provider must be given if a master_key is given')
    self._check_closed()
    with _wrap_encryption_errors():
        raw_result = self._encryption.rewrap_many_data_key(filter, provider, master_key)
        if raw_result is None:
            return RewrapManyDataKeyResult()
    raw_doc = RawBSONDocument(raw_result, DEFAULT_RAW_BSON_OPTIONS)
    replacements = []
    for key in raw_doc['v']:
        update_model = {'$set': {'keyMaterial': key['keyMaterial'], 'masterKey': key['masterKey']}, '$currentDate': {'updateDate': True}}
        op = UpdateOne({'_id': key['_id']}, update_model)
        replacements.append(op)
    if not replacements:
        return RewrapManyDataKeyResult()
    assert self._key_vault_coll is not None
    result = self._key_vault_coll.bulk_write(replacements)
    return RewrapManyDataKeyResult(result)