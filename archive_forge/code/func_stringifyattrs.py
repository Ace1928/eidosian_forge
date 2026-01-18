from fontTools.misc.textTools import byteord, strjoin, tobytes, tostr
import sys
import os
import string
def stringifyattrs(self, *args, **kwargs):
    if kwargs:
        assert not args
        attributes = sorted(kwargs.items())
    elif args:
        assert len(args) == 1
        attributes = args[0]
    else:
        return ''
    data = ''
    for attr, value in attributes:
        if not isinstance(value, (bytes, str)):
            value = str(value)
        data = data + ' %s="%s"' % (attr, escapeattr(value))
    return data