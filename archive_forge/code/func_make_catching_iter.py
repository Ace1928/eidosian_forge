import sys
import traceback
import cgi
from io import StringIO
from paste.exceptions import formatter, collector, reporter
from paste import wsgilib
from paste import request
def make_catching_iter(self, app_iter, environ, sr_checker):
    if isinstance(app_iter, (list, tuple)):
        return app_iter
    return CatchingIter(app_iter, environ, sr_checker, self)