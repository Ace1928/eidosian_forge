from __future__ import annotations
import typing as t
import zlib
from ._json import _CompactJSON
from .encoding import base64_decode
from .encoding import base64_encode
from .exc import BadPayload
from .serializer import _PDataSerializer
from .serializer import Serializer
from .timed import TimedSerializer
def load_payload(self, payload: bytes, *args: t.Any, serializer: t.Any | None=None, **kwargs: t.Any) -> t.Any:
    decompress = False
    if payload.startswith(b'.'):
        payload = payload[1:]
        decompress = True
    try:
        json = base64_decode(payload)
    except Exception as e:
        raise BadPayload('Could not base64 decode the payload because of an exception', original_error=e) from e
    if decompress:
        try:
            json = zlib.decompress(json)
        except Exception as e:
            raise BadPayload('Could not zlib decompress the payload before decoding the payload', original_error=e) from e
    return super().load_payload(json, *args, **kwargs)