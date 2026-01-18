from collections import defaultdict
from struct import pack, pack_into, unpack_from
from . import spec
from .basic_message import Message
from .exceptions import UnexpectedFrame
from .utils import str_to_bytes
def frame_handler(connection, callback, unpack_from=unpack_from, content_methods=_CONTENT_METHODS):
    """Create closure that reads frames."""
    expected_types = defaultdict(lambda: 1)
    partial_messages = {}

    def on_frame(frame):
        frame_type, channel, buf = frame
        connection.bytes_recv += 1
        if frame_type not in (expected_types[channel], 8):
            raise UnexpectedFrame('Received frame {} while expecting type: {}'.format(frame_type, expected_types[channel]))
        elif frame_type == 1:
            method_sig = unpack_from('>HH', buf, 0)
            if method_sig in content_methods:
                partial_messages[channel] = Message(frame_method=method_sig, frame_args=buf)
                expected_types[channel] = 2
                return False
            callback(channel, method_sig, buf, None)
        elif frame_type == 2:
            msg = partial_messages[channel]
            msg.inbound_header(buf)
            if not msg.ready:
                expected_types[channel] = 3
                return False
            expected_types[channel] = 1
            partial_messages.pop(channel, None)
            callback(channel, msg.frame_method, msg.frame_args, msg)
        elif frame_type == 3:
            msg = partial_messages[channel]
            msg.inbound_body(buf)
            if not msg.ready:
                return False
            expected_types[channel] = 1
            partial_messages.pop(channel, None)
            callback(channel, msg.frame_method, msg.frame_args, msg)
        elif frame_type == 8:
            return False
        return True
    return on_frame