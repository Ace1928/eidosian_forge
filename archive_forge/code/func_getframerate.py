from collections import namedtuple
import warnings
def getframerate(self):
    if not self._framerate:
        raise Error('frame rate not set')
    return self._framerate