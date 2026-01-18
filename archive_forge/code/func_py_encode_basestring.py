import re
def py_encode_basestring(s):
    """Return a JSON representation of a Python string

    """

    def replace(match):
        return ESCAPE_DCT[match.group(0)]
    return '"' + ESCAPE.sub(replace, s) + '"'