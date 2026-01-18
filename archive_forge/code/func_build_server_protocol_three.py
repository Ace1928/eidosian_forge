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
def build_server_protocol_three(backing_transport, write_func, root_client_path, jail_root=None):
    request_handler = request.SmartServerRequestHandler(backing_transport, commands=request.request_handlers, root_client_path=root_client_path, jail_root=jail_root)
    responder = ProtocolThreeResponder(write_func)
    message_handler = message.ConventionalRequestHandler(request_handler, responder)
    return ProtocolThreeDecoder(message_handler)