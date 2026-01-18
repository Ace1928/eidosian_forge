from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
def call_with_body_stream(self, args, stream):
    response, response_handler = self._call_and_read_response(args[0], args[1:], body_stream=stream, expect_response_body=False)
    return (response, response_handler)