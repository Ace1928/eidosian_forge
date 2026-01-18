from __future__ import annotations
import io
import typing
from base64 import b64encode
from enum import Enum
from ..exceptions import UnrewindableBodyError
from .util import to_bytes
def body_to_chunks(body: typing.Any | None, method: str, blocksize: int) -> ChunksAndContentLength:
    """Takes the HTTP request method, body, and blocksize and
    transforms them into an iterable of chunks to pass to
    socket.sendall() and an optional 'Content-Length' header.

    A 'Content-Length' of 'None' indicates the length of the body
    can't be determined so should use 'Transfer-Encoding: chunked'
    for framing instead.
    """
    chunks: typing.Iterable[bytes] | None
    content_length: int | None
    if body is None:
        chunks = None
        if method.upper() not in _METHODS_NOT_EXPECTING_BODY:
            content_length = 0
        else:
            content_length = None
    elif isinstance(body, (str, bytes)):
        chunks = (to_bytes(body),)
        content_length = len(chunks[0])
    elif hasattr(body, 'read'):

        def chunk_readable() -> typing.Iterable[bytes]:
            nonlocal body, blocksize
            encode = isinstance(body, io.TextIOBase)
            while True:
                datablock = body.read(blocksize)
                if not datablock:
                    break
                if encode:
                    datablock = datablock.encode('iso-8859-1')
                yield datablock
        chunks = chunk_readable()
        content_length = None
    else:
        try:
            mv = memoryview(body)
        except TypeError:
            try:
                chunks = iter(body)
                content_length = None
            except TypeError:
                raise TypeError(f"'body' must be a bytes-like object, file-like object, or iterable. Instead was {body!r}") from None
        else:
            chunks = (body,)
            content_length = mv.nbytes
    return ChunksAndContentLength(chunks=chunks, content_length=content_length)