import re
import sys
import tempfile
from urllib.parse import unquote
import cheroot.server
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
def read_lines_to_boundary(self, fp_out=None):
    """Read bytes from self.fp and return or write them to a file.

        If the 'fp_out' argument is None (the default), all bytes read are
        returned in a single byte string.

        If the 'fp_out' argument is not None, it must be a file-like
        object that supports the 'write' method; all bytes read will be
        written to the fp, and that fp is returned.
        """
    endmarker = self.boundary + b'--'
    delim = b''
    prev_lf = True
    lines = []
    seen = 0
    while True:
        line = self.fp.readline(1 << 16)
        if not line:
            raise EOFError('Illegal end of multipart body.')
        if line.startswith(b'--') and prev_lf:
            strippedline = line.strip()
            if strippedline == self.boundary:
                break
            if strippedline == endmarker:
                self.fp.finish()
                break
        line = delim + line
        if line.endswith(b'\r\n'):
            delim = b'\r\n'
            line = line[:-2]
            prev_lf = True
        elif line.endswith(b'\n'):
            delim = b'\n'
            line = line[:-1]
            prev_lf = True
        else:
            delim = b''
            prev_lf = False
        if fp_out is None:
            lines.append(line)
            seen += len(line)
            if seen > self.maxrambytes:
                fp_out = self.make_file()
                for line in lines:
                    fp_out.write(line)
        else:
            fp_out.write(line)
    if fp_out is None:
        result = b''.join(lines)
        return result
    else:
        fp_out.seek(0)
        return fp_out