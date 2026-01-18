import cgitb
from io import StringIO
import sys
from paste.util import converters
def make_cgitb_middleware(app, global_conf, display=NoDefault, logdir=None, context=5, format='html'):
    """
    Wraps the application in the ``cgitb`` (standard library)
    error catcher.

      display:
        If true (or debug is set in the global configuration)
        then the traceback will be displayed in the browser

      logdir:
        Writes logs of all errors in that directory

      context:
        Number of lines of context to show around each line of
        source code
    """
    from paste.deploy.converters import asbool
    if display is not NoDefault:
        display = asbool(display)
    if 'debug' in global_conf:
        global_conf['debug'] = asbool(global_conf['debug'])
    return CgitbMiddleware(app, global_conf=global_conf, display=display, logdir=logdir, context=context, format=format)