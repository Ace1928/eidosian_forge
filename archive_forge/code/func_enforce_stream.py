from typing import (
from urllib.parse import urlparse
def enforce_stream(value: Union[bytes, Iterable[bytes], AsyncIterable[bytes], None], *, name: str) -> Union[Iterable[bytes], AsyncIterable[bytes]]:
    if value is None:
        return ByteStream(b'')
    elif isinstance(value, bytes):
        return ByteStream(value)
    return value