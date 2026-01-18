import os
import sys
import time
import struct
import socket
import random
from eventlet.green import threading
from eventlet.zipkin._thrift.zipkinCore import ttypes
from eventlet.zipkin._thrift.zipkinCore.constants import SERVER_SEND
def put_key_value(key, value, endpoint=None):
    """ This is binary annotation API.
    You can add your own key-value extra information from in your code.
    Key-value doesn't have a time component.
    e.g.) put_key_value('http.uri', '/hoge/index.html')

    :param key: String
    :param value: String
    :param endpoint: host info
    """
    if is_sample():
        b = ZipkinDataBuilder.build_binary_annotation(key, value, endpoint)
        trace_data = get_trace_data()
        trace_data.add_binary_annotation(b)