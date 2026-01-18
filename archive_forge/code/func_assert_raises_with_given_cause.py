import contextlib
import re
import sys
def assert_raises_with_given_cause(except_cls, cause_cls, callable_, *args, **kw):
    return _assert_raises(except_cls, callable_, args, kw, cause_cls=cause_cls)