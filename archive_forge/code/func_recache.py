import os
import traceback
from io import StringIO
from twisted import copyright
from twisted.python.compat import execfile, networkString
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.web import http, resource, server, static, util
import mygreatresource
def recache(self):
    self.doCache = 1