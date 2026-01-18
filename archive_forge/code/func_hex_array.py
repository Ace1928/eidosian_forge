import importlib
import logging
import os
import sys
def hex_array(data):
    """
    Convert bytes or bytearray into array of hexes to be printed.
    """
    return ' '.join(('0x%02x' % byte for byte in bytearray(data)))