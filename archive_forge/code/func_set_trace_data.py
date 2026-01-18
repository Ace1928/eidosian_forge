import os
import sys
import time
import struct
import socket
import random
from eventlet.green import threading
from eventlet.zipkin._thrift.zipkinCore import ttypes
from eventlet.zipkin._thrift.zipkinCore.constants import SERVER_SEND
def set_trace_data(trace_data):
    _tls.trace_data = trace_data