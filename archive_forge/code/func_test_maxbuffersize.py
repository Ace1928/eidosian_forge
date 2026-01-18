import io
import pytest
from srsly.msgpack import Unpacker, BufferFull
from srsly.msgpack import pack
from srsly.msgpack.exceptions import OutOfData
def test_maxbuffersize():
    with pytest.raises(ValueError):
        Unpacker(read_size=5, max_buffer_size=3)
    unpacker = Unpacker(read_size=3, max_buffer_size=3, use_list=1)
    unpacker.feed(b'fo')
    with pytest.raises(BufferFull):
        unpacker.feed(b'ob')
    unpacker.feed(b'o')
    assert ord('f') == next(unpacker)
    unpacker.feed(b'b')
    assert ord('o') == next(unpacker)
    assert ord('o') == next(unpacker)
    assert ord('b') == next(unpacker)