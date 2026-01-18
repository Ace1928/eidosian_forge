import logging
from OpenGL import platform, _configflags
from ctypes import ArgumentError
def onEnd(self):
    """Called by glEnd to record the fact that glGetError will work"""
    self._currentChecker = self._registeredChecker