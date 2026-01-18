from OpenGL.latebind import LateBind
from OpenGL._bytes import bytes,unicode,as_8_bit
import OpenGL as root
import sys
import logging
@classmethod
def hasExtension(self, specifier):
    for registered in self.registered:
        result = registered(specifier)
        if result:
            return result
    return False