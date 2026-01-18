import re
import sys
import time
def assert_raises_message(except_cls, msg, callable_, *args, **kwargs):
    try:
        callable_(*args, **kwargs)
        assert False, 'Callable did not raise an exception'
    except except_cls as e:
        assert re.search(msg, str(e)), '%r !~ %s' % (msg, e)