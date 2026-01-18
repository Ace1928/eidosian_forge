import re
import sys
import traceback
from typing import NoReturn
import pytest
from .._util import (
def test_bytesify() -> None:
    assert bytesify(b'123') == b'123'
    assert bytesify(bytearray(b'123')) == b'123'
    assert bytesify('123') == b'123'
    with pytest.raises(UnicodeEncodeError):
        bytesify('ሴ')
    with pytest.raises(TypeError):
        bytesify(10)