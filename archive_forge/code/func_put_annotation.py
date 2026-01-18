import os
import sys
import time
import struct
import socket
import random
from eventlet.green import threading
from eventlet.zipkin._thrift.zipkinCore import ttypes
from eventlet.zipkin._thrift.zipkinCore.constants import SERVER_SEND
def put_annotation(msg, endpoint=None):
    """ This is annotation API.
    You can add your own annotation from in your code.
    Annotation is recorded with timestamp automatically.
    e.g.) put_annotation('cache hit for %s' % request)

    :param msg: String message
    :param endpoint: host info
    """
    if is_sample():
        a = ZipkinDataBuilder.build_annotation(msg, endpoint)
        trace_data = get_trace_data()
        trace_data.add_annotation(a)