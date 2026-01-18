from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
def get_next_message(self) -> Optional[Message]:
    """Parse one message, if there is enough data.

        Returns None if it doesn't have a complete message.
        """
    if self.next_msg_size is None:
        if self.buf.bytes_buffered >= 16:
            self.next_msg_size = calc_msg_size(self.buf.peek(16))
    nms = self.next_msg_size
    if nms is not None and self.buf.bytes_buffered >= nms:
        raw_msg = self.buf.read(nms)
        msg = Message.from_buffer(raw_msg, fds=self.fds)
        self.next_msg_size = None
        fds_consumed = msg.header.fields.get(HeaderFields.unix_fds, 0)
        self.fds = self.fds[fds_consumed:]
        return msg