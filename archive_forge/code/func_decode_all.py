from __future__ import annotations
import datetime
import itertools
import os
import re
import struct
import sys
import uuid
from codecs import utf_8_decode as _utf_8_decode
from codecs import utf_8_encode as _utf_8_encode
from collections import abc as _abc
from typing import (
from bson.binary import (
from bson.code import Code
from bson.codec_options import (
from bson.datetime_ms import (
from bson.dbref import DBRef
from bson.decimal128 import Decimal128
from bson.errors import InvalidBSON, InvalidDocument, InvalidStringData
from bson.int64 import Int64
from bson.max_key import MaxKey
from bson.min_key import MinKey
from bson.objectid import ObjectId
from bson.regex import Regex
from bson.son import RE_TYPE, SON
from bson.timestamp import Timestamp
from bson.tz_util import utc
def decode_all(data: _ReadableBuffer, codec_options: Optional[CodecOptions[_DocumentType]]=None) -> Union[list[dict[str, Any]], list[_DocumentType]]:
    """Decode BSON data to multiple documents.

    `data` must be a bytes-like object implementing the buffer protocol that
    provides concatenated, valid, BSON-encoded documents.

    :Parameters:
      - `data`: BSON data
      - `codec_options` (optional): An instance of
        :class:`~bson.codec_options.CodecOptions`.

    .. versionchanged:: 3.9
       Supports bytes-like objects that implement the buffer protocol.

    .. versionchanged:: 3.0
       Removed `compile_re` option: PyMongo now always represents BSON regular
       expressions as :class:`~bson.regex.Regex` objects. Use
       :meth:`~bson.regex.Regex.try_compile` to attempt to convert from a
       BSON regular expression to a Python regular expression object.

       Replaced `as_class`, `tz_aware`, and `uuid_subtype` options with
       `codec_options`.
    """
    if codec_options is None:
        return _decode_all(data, DEFAULT_CODEC_OPTIONS)
    if not isinstance(codec_options, CodecOptions):
        raise _CODEC_OPTIONS_TYPE_ERROR
    return _decode_all(data, codec_options)