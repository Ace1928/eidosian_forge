import random
from eventlet import wsgi
from eventlet.zipkin import api
from eventlet.zipkin._thrift.zipkinCore.constants import \
from eventlet.zipkin.http import \
def bool_or_none(val):
    if val == '1':
        return True
    if val == '0':
        return False
    return None