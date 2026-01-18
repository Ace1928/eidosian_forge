import struct
import time
import io
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import file_generator
from cherrypy.lib import is_closable_iterator
from cherrypy.lib import set_vary_header
def find_acceptable_charset(self):
    request = cherrypy.serving.request
    response = cherrypy.serving.response
    if self.debug:
        cherrypy.log('response.stream %r' % response.stream, 'TOOLS.ENCODE')
    if response.stream:
        encoder = self.encode_stream
    else:
        encoder = self.encode_string
        if 'Content-Length' in response.headers:
            del response.headers['Content-Length']
    encs = request.headers.elements('Accept-Charset')
    charsets = [enc.value.lower() for enc in encs]
    if self.debug:
        cherrypy.log('charsets %s' % repr(charsets), 'TOOLS.ENCODE')
    if self.encoding is not None:
        encoding = self.encoding.lower()
        if self.debug:
            cherrypy.log('Specified encoding %r' % encoding, 'TOOLS.ENCODE')
        if not charsets or '*' in charsets or encoding in charsets:
            if self.debug:
                cherrypy.log('Attempting encoding %r' % encoding, 'TOOLS.ENCODE')
            if encoder(encoding):
                return encoding
    elif not encs:
        if self.debug:
            cherrypy.log('Attempting default encoding %r' % self.default_encoding, 'TOOLS.ENCODE')
        if encoder(self.default_encoding):
            return self.default_encoding
        else:
            raise cherrypy.HTTPError(500, self.failmsg % self.default_encoding)
    else:
        for element in encs:
            if element.qvalue > 0:
                if element.value == '*':
                    if self.debug:
                        cherrypy.log('Attempting default encoding due to %r' % element, 'TOOLS.ENCODE')
                    if encoder(self.default_encoding):
                        return self.default_encoding
                else:
                    encoding = element.value
                    if self.debug:
                        cherrypy.log('Attempting encoding %s (qvalue >0)' % element, 'TOOLS.ENCODE')
                    if encoder(encoding):
                        return encoding
        if '*' not in charsets:
            iso = 'iso-8859-1'
            if iso not in charsets:
                if self.debug:
                    cherrypy.log('Attempting ISO-8859-1 encoding', 'TOOLS.ENCODE')
                if encoder(iso):
                    return iso
    ac = request.headers.get('Accept-Charset')
    if ac is None:
        msg = 'Your client did not send an Accept-Charset header.'
    else:
        msg = 'Your client sent this Accept-Charset header: %s.' % ac
    _charsets = ', '.join(sorted(self.attempted_charsets))
    msg += ' We tried these charsets: %s.' % (_charsets,)
    raise cherrypy.HTTPError(406, msg)