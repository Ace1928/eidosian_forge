import contextlib
import re
import sys
def assert_raises_message_with_proper_context(except_cls, msg, callable_, *args, **kwargs):
    return _assert_raises(except_cls, callable_, args, kwargs, msg=msg, check_context=True)