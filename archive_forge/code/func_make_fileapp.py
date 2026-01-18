import mimetypes
import os
from webob import exc
from webob.dec import wsgify
from webob.response import Response
def make_fileapp(self, path):
    return FileApp(path, **self.fileapp_kw)