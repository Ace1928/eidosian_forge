from typing import Dict, Tuple, TypeVar
from google.protobuf import any_pb2
from google.protobuf.message import Message
import cirq
def pack_any(message: Message) -> any_pb2.Any:
    """Packs a message into an Any proto.

    Returns the packed Any proto.
    """
    packed = any_pb2.Any()
    packed.Pack(message)
    return packed