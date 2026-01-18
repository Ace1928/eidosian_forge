import os
from typing import SupportsBytes, Type, TypeVar
import nacl.bindings
from nacl import encoding
def randombytes_deterministic(size: int, seed: bytes, encoder: encoding.Encoder=encoding.RawEncoder) -> bytes:
    """
    Returns ``size`` number of deterministically generated pseudorandom bytes
    from a seed

    :param size: int
    :param seed: bytes
    :param encoder: The encoder class used to encode the produced bytes
    :rtype: bytes
    """
    raw_data = nacl.bindings.randombytes_buf_deterministic(size, seed)
    return encoder.encode(raw_data)