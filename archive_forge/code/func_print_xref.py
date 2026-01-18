import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def print_xref(doc, xref):
    """Print an object given by XREF number.

    Simulate the PDF source in "pretty" format.
    For a stream also print its size.
    """
    fitz.message('%i 0 obj' % xref)
    xref_str = doc.xref_object(xref)
    fitz.message(xref_str)
    if doc.xref_is_stream(xref):
        temp = xref_str.split()
        try:
            idx = temp.index('/Length') + 1
            size = temp[idx]
            if size.endswith('0 R'):
                size = 'unknown'
        except Exception:
            size = 'unknown'
        fitz.message('stream\n...%s bytes' % size)
        fitz.message('endstream')
    fitz.message('endobj')