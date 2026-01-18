from typing import (
from urllib.parse import urlparse
def include_request_headers(headers: List[Tuple[bytes, bytes]], *, url: 'URL', content: Union[None, bytes, Iterable[bytes], AsyncIterable[bytes]]) -> List[Tuple[bytes, bytes]]:
    headers_set = set((k.lower() for k, v in headers))
    if b'host' not in headers_set:
        default_port = DEFAULT_PORTS.get(url.scheme)
        if url.port is None or url.port == default_port:
            header_value = url.host
        else:
            header_value = b'%b:%d' % (url.host, url.port)
        headers = [(b'Host', header_value)] + headers
    if content is not None and b'content-length' not in headers_set and (b'transfer-encoding' not in headers_set):
        if isinstance(content, bytes):
            content_length = str(len(content)).encode('ascii')
            headers += [(b'Content-Length', content_length)]
        else:
            headers += [(b'Transfer-Encoding', b'chunked')]
    return headers