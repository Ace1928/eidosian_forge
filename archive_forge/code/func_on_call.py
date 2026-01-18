from .abstract import Thenable
from .promises import promise
def on_call(*args, **kwargs):
    if len(args) == 1 and isinstance(args[0], promise):
        return args[0].then(p)
    else:
        return p(*args, **kwargs)