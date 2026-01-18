import asyncio
from contextlib import suppress
from typing import Any, Optional, Tuple
from .base_protocol import BaseProtocol
from .client_exceptions import (
from .helpers import BaseTimerContext, status_code_must_be_empty_body
from .http import HttpResponseParser, RawResponseMessage
from .streams import EMPTY_PAYLOAD, DataQueue, StreamReader
def set_response_params(self, *, timer: Optional[BaseTimerContext]=None, skip_payload: bool=False, read_until_eof: bool=False, auto_decompress: bool=True, read_timeout: Optional[float]=None, read_bufsize: int=2 ** 16, timeout_ceil_threshold: float=5, max_line_size: int=8190, max_field_size: int=8190) -> None:
    self._skip_payload = skip_payload
    self._read_timeout = read_timeout
    self._timeout_ceil_threshold = timeout_ceil_threshold
    self._parser = HttpResponseParser(self, self._loop, read_bufsize, timer=timer, payload_exception=ClientPayloadError, response_with_body=not skip_payload, read_until_eof=read_until_eof, auto_decompress=auto_decompress, max_line_size=max_line_size, max_field_size=max_field_size)
    if self._tail:
        data, self._tail = (self._tail, b'')
        self.data_received(data)