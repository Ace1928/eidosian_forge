import sys
import re
from urllib.parse import urlparse
def parse_paths(uri, vfs=None):
    """Parse a URI or Apache VFS URL into its parts

    Returns: tuple
        (path, scheme, archive)
    """
    archive = scheme = None
    path = uri
    if sys.platform == 'win32' and re.match('^[a-zA-Z]\\:', path):
        return (path, None, None)
    if vfs:
        parts = urlparse(vfs)
        scheme = parts.scheme
        archive = parts.path
        if parts.netloc and parts.netloc != 'localhost':
            archive = parts.netloc + archive
    else:
        parts = urlparse(path)
        scheme = parts.scheme
        path = parts.path
        if parts.netloc and parts.netloc != 'localhost':
            if scheme.split('+')[-1] in CURLSCHEMES:
                path = '{}://{}{}'.format(scheme.split('+')[-1], parts.netloc, path)
            else:
                path = parts.netloc + path
        if scheme in SCHEMES:
            parts = path.split('!')
            path = parts.pop() if parts else None
            archive = parts.pop() if parts else None
    scheme = None if not scheme else scheme
    return (path, scheme, archive)