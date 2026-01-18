import functools
import types
from fixtures import Fixture
@functools.wraps(old_value)
def avoid_get(*args, **kwargs):
    return captured_method(*args, **kwargs)