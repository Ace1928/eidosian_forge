from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers
import sys, os, time
def read_environ():
    """Read environment, fixing HTTP variables"""
    enc = sys.getfilesystemencoding()
    esc = 'surrogateescape'
    try:
        ''.encode('utf-8', esc)
    except LookupError:
        esc = 'replace'
    environ = {}
    for k, v in os.environ.items():
        if _needs_transcode(k):
            if sys.platform == 'win32':
                software = os.environ.get('SERVER_SOFTWARE', '').lower()
                if software.startswith('microsoft-iis/'):
                    v = v.encode('utf-8').decode('iso-8859-1')
                elif software.startswith('apache/'):
                    pass
                elif software.startswith('simplehttp/') and 'python/3' in software:
                    v = v.encode('utf-8').decode('iso-8859-1')
                else:
                    v = v.encode(enc, 'replace').decode('iso-8859-1')
            else:
                v = v.encode(enc, esc).decode('iso-8859-1')
        environ[k] = v
    return environ