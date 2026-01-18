import re
import sys
import tempfile
from urllib.parse import unquote
import cheroot.server
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
def process_multipart(entity):
    """Read all multipart parts into entity.parts."""
    ib = ''
    if 'boundary' in entity.content_type.params:
        ib = entity.content_type.params['boundary'].strip('"')
    if not re.match('^[ -~]{0,200}[!-~]$', ib):
        raise ValueError('Invalid boundary in multipart form: %r' % (ib,))
    ib = ('--' + ib).encode('ascii')
    while True:
        b = entity.readline()
        if not b:
            return
        b = b.strip()
        if b == ib:
            break
    while True:
        part = entity.part_class.from_fp(entity.fp, ib)
        entity.parts.append(part)
        part.process()
        if part.fp.done:
            break