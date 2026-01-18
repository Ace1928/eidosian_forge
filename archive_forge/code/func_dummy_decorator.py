import functools
from oslotest import base as test_base
from oslo_utils import reflection
def dummy_decorator(f):

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper