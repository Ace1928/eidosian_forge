from .structures import LookupDict
def doc(code):
    names = ', '.join(('``%s``' % n for n in _codes[code]))
    return '* %d: %s' % (code, names)