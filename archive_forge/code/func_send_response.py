import _thread
import struct
import sys
import time
from collections import deque
from io import BytesIO
from fastbencode import bdecode_as_tuple, bencode
import breezy
from ... import debug, errors, osutils
from ...trace import log_exception_quietly, mutter
from . import message, request
def send_response(self, response):
    if self.response_sent:
        raise AssertionError('send_response(%r) called, but response already sent.' % (response,))
    self.response_sent = True
    self._write_protocol_version()
    self._write_headers(self._headers)
    if response.is_successful():
        self._write_success_status()
    else:
        self._write_error_status()
    if 'hpss' in debug.debug_flags:
        self._trace('response', repr(response.args))
    self._write_structure(response.args)
    if response.body is not None:
        self._write_prefixed_body(response.body)
        if 'hpss' in debug.debug_flags:
            self._trace('body', '%d bytes' % (len(response.body),), response.body, include_time=True)
    elif response.body_stream is not None:
        count = num_bytes = 0
        first_chunk = None
        for exc_info, chunk in _iter_with_errors(response.body_stream):
            count += 1
            if exc_info is not None:
                self._write_error_status()
                error_struct = request._translate_error(exc_info[1])
                self._write_structure(error_struct)
                break
            else:
                if isinstance(chunk, request.FailedSmartServerResponse):
                    self._write_error_status()
                    self._write_structure(chunk.args)
                    break
                num_bytes += len(chunk)
                if first_chunk is None:
                    first_chunk = chunk
                self._write_prefixed_body(chunk)
                self.flush()
                if 'hpssdetail' in debug.debug_flags:
                    self._trace('body chunk', '%d bytes' % (len(chunk),), chunk, suppress_time=True)
        if 'hpss' in debug.debug_flags:
            self._trace('body stream', '%d bytes %d chunks' % (num_bytes, count), first_chunk)
    self._write_end()
    if 'hpss' in debug.debug_flags:
        self._trace('response end', '', include_time=True)