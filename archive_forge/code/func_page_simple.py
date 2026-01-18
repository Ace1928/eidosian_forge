import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def page_simple(page, textout, GRID, fontsize, noformfeed, skip_empty, flags):
    eop = b'\n' if noformfeed else bytes([12])
    text = page.get_text('text', flags=flags)
    if not text:
        if not skip_empty:
            textout.write(eop)
        return
    textout.write(text.encode('utf8', errors='surrogatepass'))
    textout.write(eop)
    return