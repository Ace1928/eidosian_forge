import contextlib
import re
import sys
def expect_raises_message_with_proper_context(except_cls, msg, check_context=True):
    return _expect_raises(except_cls, msg=msg, check_context=check_context)