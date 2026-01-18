import base64
import binascii
import json
import re
import uuid
import warnings
import zlib
from collections import deque
from types import TracebackType
from typing import (
from urllib.parse import parse_qsl, unquote, urlencode
from multidict import CIMultiDict, CIMultiDictProxy, MultiMapping
from .compression_utils import ZLibCompressor, ZLibDecompressor
from .hdrs import (
from .helpers import CHAR, TOKEN, parse_mimetype, reify
from .http import HeadersParser
from .payload import (
from .streams import StreamReader
def append_form(self, obj: Union[Sequence[Tuple[str, str]], Mapping[str, str]], headers: Optional[MultiMapping[str]]=None) -> Payload:
    """Helper to append form urlencoded part."""
    assert isinstance(obj, (Sequence, Mapping))
    if headers is None:
        headers = CIMultiDict()
    if isinstance(obj, Mapping):
        obj = list(obj.items())
    data = urlencode(obj, doseq=True)
    return self.append_payload(StringPayload(data, headers=headers, content_type='application/x-www-form-urlencoded'))