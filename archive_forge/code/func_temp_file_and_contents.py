from tempfile import TemporaryFile
import threading
import pytest
from jeepney import (
from jeepney.io.threading import open_dbus_connection, DBusRouter, Proxy
@pytest.fixture()
def temp_file_and_contents():
    data = b'abc123'
    with TemporaryFile('w+b') as tf:
        tf.write(data)
        tf.flush()
        tf.seek(0)
        yield (tf, data)