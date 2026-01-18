import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def recoverpix(doc, item):
    """Return image for a given XREF."""
    x = item[0]
    s = item[1]
    if s == 0:
        return doc.extract_image(x)

    def getimage(pix):
        if pix.colorspace.n != 4:
            return pix
        tpix = fitz.Pixmap(fitz.csRGB, pix)
        return tpix
    pix1 = fitz.Pixmap(doc, x)
    pix2 = fitz.Pixmap(doc, s)
    'Sanity check:\n    - both pixmaps must have the same rectangle\n    - both pixmaps must have alpha=0\n    - pix2 must consist of 1 byte per pixel\n    '
    if not (pix1.irect == pix2.irect and pix1.alpha == pix2.alpha == 0 and (pix2.n == 1)):
        fitz.message('Warning: unsupported /SMask %i for %i:' % (s, x))
        fitz.message(pix2)
        pix2 = None
        return getimage(pix1)
    pix = fitz.Pixmap(pix1)
    pix.set_alpha(pix2.samples)
    pix1 = pix2 = None
    return getimage(pix)