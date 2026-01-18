from typing import (
from urllib.parse import urlparse
def enforce_headers(value: Union[HeadersAsMapping, HeadersAsSequence, None]=None, *, name: str) -> List[Tuple[bytes, bytes]]:
    """
    Convienence function that ensure all items in request or response headers
    are either bytes or strings in the plain ASCII range.
    """
    if value is None:
        return []
    elif isinstance(value, Mapping):
        return [(enforce_bytes(k, name='header name'), enforce_bytes(v, name='header value')) for k, v in value.items()]
    elif isinstance(value, Sequence):
        return [(enforce_bytes(k, name='header name'), enforce_bytes(v, name='header value')) for k, v in value]
    seen_type = type(value).__name__
    raise TypeError(f'{name} must be a mapping or sequence of two-tuples, but got {seen_type}.')