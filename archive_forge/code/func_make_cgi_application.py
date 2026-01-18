import os
import sys
import subprocess
from urllib.parse import quote
from paste.util import converters
def make_cgi_application(global_conf, script, path=None, include_os_environ=None, query_string=None):
    """
    Paste Deploy interface for :class:`CGIApplication`

    This object acts as a proxy to a CGI application.  You pass in the
    script path (``script``), an optional path to search for the
    script (if the name isn't absolute) (``path``).  If you don't give
    a path, then ``$PATH`` will be used.
    """
    if path is None:
        path = global_conf.get('path') or global_conf.get('PATH')
    include_os_environ = converters.asbool(include_os_environ)
    return CGIApplication(None, script, path=path, include_os_environ=include_os_environ, query_string=query_string)