import contextlib
import re
import sys
def assert_raises_with_proper_context(except_cls, callable_, *args, **kw):
    return _assert_raises(except_cls, callable_, args, kw, check_context=True)