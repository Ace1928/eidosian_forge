from __future__ import absolute_import
from base64 import b64encode
from ..exceptions import UnrewindableBodyError
from ..packages.six import b, integer_types
def set_file_position(body, pos):
    """
    If a position is provided, move file to that point.
    Otherwise, we'll attempt to record a position for future use.
    """
    if pos is not None:
        rewind_body(body, pos)
    elif getattr(body, 'tell', None) is not None:
        try:
            pos = body.tell()
        except (IOError, OSError):
            pos = _FAILEDTELL
    return pos