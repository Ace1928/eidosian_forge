import inspect
import os
import sys
def force_utf8(s, errors='strict'):
    """Same as force_bytes(s, "utf8", errors)"""
    return force_bytes(s, 'utf8', errors)