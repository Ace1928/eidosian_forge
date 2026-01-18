import pytest
from pathlib import Path
import datetime
from mock import patch
import numpy
from .._msgpack_api import read_msgpack, write_msgpack
from .._msgpack_api import msgpack_loads, msgpack_dumps
from .._msgpack_api import msgpack_encoders, msgpack_decoders
from .util import make_tempdir
def test_msgpack_dumps():
    data = {'hello': 'world', 'test': 123}
    expected = [b'\x82\xa5hello\xa5world\xa4test{', b'\x82\xa4test{\xa5hello\xa5world']
    msg = msgpack_dumps(data)
    assert msg in expected