from contextlib import contextmanager
from itertools import count
from jeepney import HeaderFields, Message, MessageFlag, MessageType
def check_replyable(msg: Message):
    """Raise an error if we wouldn't expect a reply for msg"""
    if msg.header.message_type != MessageType.method_call:
        raise TypeError(f'Only method call messages have replies (not {msg.header.message_type})')
    if MessageFlag.no_reply_expected & msg.header.flags:
        raise ValueError('This message has the no_reply_expected flag set')