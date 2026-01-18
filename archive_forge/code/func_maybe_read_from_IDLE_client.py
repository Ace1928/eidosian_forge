import re
from typing import Any, Callable, Dict, Iterable, NoReturn, Optional, Tuple, Type, Union
from ._abnf import chunk_header, header_field, request_line, status_line
from ._events import Data, EndOfMessage, InformationalResponse, Request, Response
from ._receivebuffer import ReceiveBuffer
from ._state import (
from ._util import LocalProtocolError, RemoteProtocolError, Sentinel, validate
def maybe_read_from_IDLE_client(buf: ReceiveBuffer) -> Optional[Request]:
    lines = buf.maybe_extract_lines()
    if lines is None:
        if buf.is_next_line_obviously_invalid_request_line():
            raise LocalProtocolError('illegal request line')
        return None
    if not lines:
        raise LocalProtocolError('no request line received')
    matches = validate(request_line_re, lines[0], 'illegal request line: {!r}', lines[0])
    return Request(headers=list(_decode_header_lines(lines[1:])), _parsed=True, **matches)