import os
import sys
import time
import struct
import socket
import random
from eventlet.green import threading
from eventlet.zipkin._thrift.zipkinCore import ttypes
from eventlet.zipkin._thrift.zipkinCore.constants import SERVER_SEND
@staticmethod
def build_annotation(value, endpoint=None):
    if isinstance(value, str):
        value = value.encode('utf-8')
    assert isinstance(value, bytes)
    return ttypes.Annotation(time.time() * 1000 * 1000, value, endpoint)