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
def remove_key_alt_name(self, id: Binary, key_alt_name: str) -> Optional[RawBSONDocument]:
    """Remove ``key_alt_name`` from the set of keyAltNames in the key document with UUID ``id``.

        Also removes the ``keyAltNames`` field from the key document if it would otherwise be empty.

        :Parameters:
          - ``id``: The UUID of a key a which must be a
            :class:`~bson.binary.Binary` with subtype 4 (
            :attr:`~bson.binary.UUID_SUBTYPE`).
          - ``key_alt_name``: The key alternate name to remove.

        :Returns:
          Returns the previous version of the key document.

        .. versionadded:: 4.2
        """
    self._check_closed()
    pipeline = [{'$set': {'keyAltNames': {'$cond': [{'$eq': ['$keyAltNames', [key_alt_name]]}, '$$REMOVE', {'$filter': {'input': '$keyAltNames', 'cond': {'$ne': ['$$this', key_alt_name]}}}]}}}]
    assert self._key_vault_coll is not None
    return self._key_vault_coll.find_one_and_update({'_id': id}, pipeline)