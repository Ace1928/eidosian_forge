from __future__ import annotations
import contextlib
import enum
import errno
import hmac
import os
import struct
import warnings
import weakref
from itertools import count
from typing import (
from weakref import ReferenceType, WeakValueDictionary
import attrs
import trio
from ._util import NoPublicConstructor, final
def challenge_for(key: bytes, address: Any, epoch_seqno: int, client_hello_bits: bytes) -> bytes:
    salt = os.urandom(SALT_BYTES)
    tick = _current_cookie_tick()
    cookie = _make_cookie(key, salt, tick, address, client_hello_bits)
    body = ProtocolVersion.DTLS10 + bytes([len(cookie)]) + cookie
    hs = HandshakeFragment(msg_type=HandshakeType.hello_verify_request, msg_len=len(body), msg_seq=0, frag_offset=0, frag_len=len(body), frag=body)
    payload = encode_handshake_fragment(hs)
    packet = encode_record(Record(ContentType.handshake, ProtocolVersion.DTLS10, epoch_seqno, payload))
    return packet