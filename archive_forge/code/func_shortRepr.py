import logging
from OpenGL import platform, _configflags
from ctypes import ArgumentError
def shortRepr(self, value, firstLevel=True):
    """Retrieve short representation of the given value"""
    if isinstance(value, (list, tuple)) and value and (len(repr(value)) >= 40):
        if isinstance(value, list):
            template = '[\n\t\t%s\n\t]'
        else:
            template = '(\n\t\t%s,\n\t)'
        return template % ',\n\t\t'.join([self.shortRepr(x, False) for x in value])
    r = repr(value)
    if len(r) < 120:
        return r
    else:
        return r[:117] + '...'