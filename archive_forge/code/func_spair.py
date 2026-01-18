import signal
import socket
import threading
import time
import pytest
from urllib3.util.wait import (
from .socketpair_helper import socketpair
@pytest.fixture
def spair():
    a, b = socketpair()
    yield (a, b)
    a.close()
    b.close()