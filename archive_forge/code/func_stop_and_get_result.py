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
def stop_and_get_result(self):
    import struct
    self.sink.write(struct.pack('Q', 0))
    self.sink.flush()
    self._sock.close()
    self._server.join()
    return self._server.get_result()