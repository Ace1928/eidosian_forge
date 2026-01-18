import struct
import time
import io
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import file_generator
from cherrypy.lib import is_closable_iterator
from cherrypy.lib import set_vary_header
def prepare_iter(value):
    """
    Ensure response body is iterable and resolves to False when empty.
    """
    if isinstance(value, text_or_bytes):
        if value:
            value = [value]
        else:
            value = []
    elif hasattr(value, 'read'):
        value = file_generator(value)
    elif value is None:
        value = []
    return value