import sys
def coerce_text(v):
    if not isinstance(v, basestring_):
        if sys.version < '3':
            attr = '__unicode__'
        else:
            attr = '__str__'
        if hasattr(v, attr):
            return unicode(v)
        else:
            return bytes(v)
    return v