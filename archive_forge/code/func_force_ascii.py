import inspect
import os
import sys
def force_ascii(s, errors='strict'):
    """Same as force_bytes(s, "ascii", errors)"""
    return force_bytes(s, 'ascii', errors)