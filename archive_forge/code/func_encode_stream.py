import struct
import time
import io
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import file_generator
from cherrypy.lib import is_closable_iterator
from cherrypy.lib import set_vary_header
def encode_stream(self, encoding):
    """Encode a streaming response body.

        Use a generator wrapper, and just pray it works as the stream is
        being written out.
        """
    if encoding in self.attempted_charsets:
        return False
    self.attempted_charsets.add(encoding)

    def encoder(body):
        for chunk in body:
            if isinstance(chunk, str):
                chunk = chunk.encode(encoding, self.errors)
            yield chunk
    self.body = encoder(self.body)
    return True