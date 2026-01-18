import warnings
from eventlet.green import httplib
from eventlet.zipkin import api
def hex_str(n):
    """
    Thrift uses a binary representation of trace and span ids
    HTTP headers use a hexadecimal representation of the same
    """
    return '%0.16x' % (n,)