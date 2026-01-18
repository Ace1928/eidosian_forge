import logging
from OpenGL import platform, _configflags
from ctypes import ArgumentError
def nullGetError(self):
    """Used as error-checker when no error checking should be done"""
    return self._noErrorResult