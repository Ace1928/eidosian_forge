import io
import sys
import traceback
from . import util
from functools import wraps
def failfast(method):

    @wraps(method)
    def inner(self, *args, **kw):
        if getattr(self, 'failfast', False):
            self.stop()
        return method(self, *args, **kw)
    return inner