from IPython.core.error import TryNext
from functools import singledispatch
@singledispatch
def inspect_object(obj):
    """Called when you do obj?"""
    raise TryNext