import re
import sys
import traceback
from typing import NoReturn
import pytest
from .._util import (
def test_ProtocolError() -> None:
    with pytest.raises(TypeError):
        ProtocolError('abstract base class')