from promise import Promise
from .utils import assert_exception
from threading import Event
def b_then(v):
    c = Promise.resolve(None)
    d = c.then(lambda v: Promise.resolve('B'))
    return d