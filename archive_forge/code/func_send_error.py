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
def send_error(self, exception):
    if self.response_sent:
        raise AssertionError('send_error(%s) called, but response already sent.' % (exception,))
    if isinstance(exception, errors.UnknownSmartMethod):
        failure = request.FailedSmartServerResponse((b'UnknownMethod', exception.verb))
        self.send_response(failure)
        return
    if 'hpss' in debug.debug_flags:
        self._trace('error', str(exception))
    self.response_sent = True
    self._write_protocol_version()
    self._write_headers(self._headers)
    self._write_error_status()
    self._write_structure((b'error', str(exception).encode('utf-8', 'replace')))
    self._write_end()