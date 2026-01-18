import collections
import collections.abc
import copy
import inspect
from cloudsdk.google.protobuf import field_mask_pb2
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import wrappers_pb2
def from_any_pb(pb_type, any_pb):
    """Converts an ``Any`` protobuf to the specified message type.

    Args:
        pb_type (type): the type of the message that any_pb stores an instance
            of.
        any_pb (google.protobuf.any_pb2.Any): the object to be converted.

    Returns:
        pb_type: An instance of the pb_type message.

    Raises:
        TypeError: if the message could not be converted.
    """
    msg = pb_type()
    if callable(getattr(pb_type, 'pb', None)):
        msg_pb = pb_type.pb(msg)
    else:
        msg_pb = msg
    if not any_pb.Unpack(msg_pb):
        raise TypeError('Could not convert {} to {}'.format(any_pb.__class__.__name__, pb_type.__name__))
    return msg