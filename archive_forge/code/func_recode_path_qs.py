import sys as _sys
import io
import cherrypy as _cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cperror
from cherrypy.lib import httputil
from cherrypy.lib import is_closable_iterator
def recode_path_qs(self, path, qs):
    old_enc = self.environ.get('wsgi.url_encoding', 'ISO-8859-1')
    new_enc = self.cpapp.find_config(self.environ.get('PATH_INFO', ''), 'request.uri_encoding', 'utf-8')
    if new_enc.lower() == old_enc.lower():
        return
    try:
        return (path.encode(old_enc).decode(new_enc), qs.encode(old_enc).decode(new_enc))
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass