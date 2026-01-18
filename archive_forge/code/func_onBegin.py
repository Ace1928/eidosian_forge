import logging
from OpenGL import platform, _configflags
from ctypes import ArgumentError
def onBegin(self):
    """Called by glBegin to record the fact that glGetError won't work"""
    self._currentChecker = self.nullGetError