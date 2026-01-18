import io
import pytest
from srsly.msgpack import Unpacker, BufferFull
from srsly.msgpack import pack
from srsly.msgpack.exceptions import OutOfData
def test_issue124():
    unpacker = Unpacker()
    unpacker.feed(b'\xa1?\xa1!')
    assert tuple(unpacker) == (b'?', b'!')
    assert tuple(unpacker) == ()
    unpacker.feed(b'\xa1?\xa1')
    assert tuple(unpacker) == (b'?',)
    assert tuple(unpacker) == ()
    unpacker.feed(b'!')
    assert tuple(unpacker) == (b'!',)
    assert tuple(unpacker) == ()