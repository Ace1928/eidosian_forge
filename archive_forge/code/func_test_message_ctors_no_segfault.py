from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
def test_message_ctors_no_segfault():
    with pytest.raises(TypeError):
        repr(pa.Message())
    with pytest.raises(TypeError):
        repr(pa.MessageReader())