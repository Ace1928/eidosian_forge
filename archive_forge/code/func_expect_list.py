from io import StringIO, BytesIO
import codecs
import os
import sys
import re
import errno
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .expect import Expecter, searcher_string, searcher_re
def expect_list(self, pattern_list, timeout=-1, searchwindowsize=-1, async_=False, **kw):
    """This takes a list of compiled regular expressions and returns the
        index into the pattern_list that matched the child output. The list may
        also contain EOF or TIMEOUT(which are not compiled regular
        expressions). This method is similar to the expect() method except that
        expect_list() does not recompile the pattern list on every call. This
        may help if you are trying to optimize for speed, otherwise just use
        the expect() method.  This is called by expect().


        Like :meth:`expect`, passing ``async_=True`` will make this return an
        asyncio coroutine.
        """
    if timeout == -1:
        timeout = self.timeout
    if 'async' in kw:
        async_ = kw.pop('async')
    if kw:
        raise TypeError('Unknown keyword arguments: {}'.format(kw))
    exp = Expecter(self, searcher_re(pattern_list), searchwindowsize)
    if async_:
        from ._async import expect_async
        return expect_async(exp, timeout)
    else:
        return exp.expect_loop(timeout)