import eventlet
from eventlet.hubs import get_hub
def get_fileno(obj):
    try:
        f = obj.fileno
    except AttributeError:
        if not isinstance(obj, int):
            raise TypeError('Expected int or long, got %s' % type(obj))
        return obj
    else:
        rv = f()
        if not isinstance(rv, int):
            raise TypeError('Expected int or long, got %s' % type(rv))
        return rv