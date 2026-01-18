from __future__ import absolute_import
from serial.serialutil import *
def detect_java_comm(names):
    """try given list of modules and return that imports"""
    for name in names:
        try:
            mod = my_import(name)
            mod.SerialPort
            return mod
        except (ImportError, AttributeError):
            pass
    raise ImportError('No Java Communications API implementation found')