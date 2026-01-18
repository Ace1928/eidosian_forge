from .messages import Message
from .midifiles import MetaMessage, UnknownMetaMessage
def freeze_message(msg):
    """Freeze message.

    Returns a frozen version of the message. Frozen messages are
    immutable, hashable and can be used as dictionary keys.

    Will return None if called with None. This allows you to do things
    like::

        msg = freeze_message(port.poll())
    """
    if isinstance(msg, Frozen):
        return msg
    elif isinstance(msg, Message):
        class_ = FrozenMessage
    elif isinstance(msg, UnknownMetaMessage):
        class_ = FrozenUnknownMetaMessage
    elif isinstance(msg, MetaMessage):
        class_ = FrozenMetaMessage
    elif msg is None:
        return None
    else:
        raise ValueError('first argument must be a message or None')
    frozen = class_.__new__(class_)
    vars(frozen).update(vars(msg))
    return frozen