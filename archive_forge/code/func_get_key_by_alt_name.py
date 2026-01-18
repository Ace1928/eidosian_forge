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
def get_key_by_alt_name(self, key_alt_name: str) -> Optional[RawBSONDocument]:
    """Get a key document in the key vault collection that has the given ``key_alt_name``.

        :Parameters:
          - `key_alt_name`: (str): The key alternate name of the key to get.

        :Returns:
          The key document.

        .. versionadded:: 4.2
        """
    self._check_closed()
    assert self._key_vault_coll is not None
    return self._key_vault_coll.find_one({'keyAltNames': key_alt_name})