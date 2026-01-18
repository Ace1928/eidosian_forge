import re
from io import BytesIO
from .. import errors
def bytes_header(self, length, names):
    """Return the header for a Bytes record."""
    byte_sections = [b'B']
    byte_sections.append(b'%d\n' % (length,))
    for name_tuple in names:
        for name in name_tuple:
            _check_name(name)
        byte_sections.append(b'\x00'.join(name_tuple) + b'\n')
    byte_sections.append(b'\n')
    return b''.join(byte_sections)