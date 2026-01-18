from __future__ import annotations
import sys
import traceback
def set_default_encoding_file(file):
    """Set file used to get codec information."""
    global default_encoding_file
    default_encoding_file = file