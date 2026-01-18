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
def gen_list_name() -> Generator[bytes, None, None]:
    """Generate "keys" for encoded lists in the sequence
    b"0\x00", b"1\x00", b"2\x00", ...

    The first 1000 keys are returned from a pre-built cache. All
    subsequent keys are generated on the fly.
    """
    yield from _LIST_NAMES
    counter = itertools.count(1000)
    while True:
        yield (str(next(counter)) + '\x00').encode('utf8')