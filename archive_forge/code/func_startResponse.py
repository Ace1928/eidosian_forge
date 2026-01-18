from collections.abc import Sequence
from sys import exc_info
from warnings import warn
from zope.interface import implementer
from twisted.internet.threads import blockingCallFromThread
from twisted.logger import Logger
from twisted.python.failure import Failure
from twisted.web.http import INTERNAL_SERVER_ERROR
from twisted.web.resource import IResource
from twisted.web.server import NOT_DONE_YET
def startResponse(self, status, headers, excInfo=None):
    """
        The WSGI I{start_response} callable.  The given values are saved until
        they are needed to generate the response.

        This will be called in a non-I/O thread.
        """
    if self.started and excInfo is not None:
        raise excInfo[1].with_traceback(excInfo[2])
    if not isinstance(status, str):
        raise TypeError('status must be str, not {!r} ({})'.format(status, type(status).__name__))
    if isinstance(headers, list):
        pass
    elif isinstance(headers, Sequence):
        warn('headers should be a list, not %r (%s)' % (headers, type(headers).__name__), category=RuntimeWarning)
    else:
        raise TypeError('headers must be a list, not %r (%s)' % (headers, type(headers).__name__))
    for header in headers:
        if isinstance(header, tuple):
            pass
        elif isinstance(header, Sequence):
            warn('header should be a (str, str) tuple, not %r (%s)' % (header, type(header).__name__), category=RuntimeWarning)
        else:
            raise TypeError('header must be a (str, str) tuple, not %r (%s)' % (header, type(header).__name__))
        if len(header) != 2:
            raise TypeError(f'header must be a (str, str) tuple, not {header!r}')
        for elem in header:
            if not isinstance(elem, str):
                raise TypeError(f'header must be (str, str) tuple, not {header!r}')
    self.status = status
    self.headers = headers
    return self.write