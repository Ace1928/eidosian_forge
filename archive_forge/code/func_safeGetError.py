import logging
from OpenGL import platform, _configflags
from ctypes import ArgumentError
def safeGetError(self):
    """Check for error, testing for context before operation"""
    if self._isValid():
        return self._getErrors()
    return None