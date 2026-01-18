from io import StringIO, BytesIO
import codecs
import os
import sys
import re
import errno
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .expect import Expecter, searcher_string, searcher_re
def expect_exact(self, pattern_list, timeout=-1, searchwindowsize=-1, async_=False, **kw):
    """This is similar to expect(), but uses plain string matching instead
        of compiled regular expressions in 'pattern_list'. The 'pattern_list'
        may be a string; a list or other sequence of strings; or TIMEOUT and
        EOF.

        This call might be faster than expect() for two reasons: string
        searching is faster than RE matching and it is possible to limit the
        search to just the end of the input buffer.

        This method is also useful when you don't want to have to worry about
        escaping regular expression characters that you want to match.

        Like :meth:`expect`, passing ``async_=True`` will make this return an
        asyncio coroutine.
        """
    if timeout == -1:
        timeout = self.timeout
    if 'async' in kw:
        async_ = kw.pop('async')
    if kw:
        raise TypeError('Unknown keyword arguments: {}'.format(kw))
    if isinstance(pattern_list, self.allowed_string_types) or pattern_list in (TIMEOUT, EOF):
        pattern_list = [pattern_list]

    def prepare_pattern(pattern):
        if pattern in (TIMEOUT, EOF):
            return pattern
        if isinstance(pattern, self.allowed_string_types):
            return self._coerce_expect_string(pattern)
        self._pattern_type_err(pattern)
    try:
        pattern_list = iter(pattern_list)
    except TypeError:
        self._pattern_type_err(pattern_list)
    pattern_list = [prepare_pattern(p) for p in pattern_list]
    exp = Expecter(self, searcher_string(pattern_list), searchwindowsize)
    if async_:
        from ._async import expect_async
        return expect_async(exp, timeout)
    else:
        return exp.expect_loop(timeout)