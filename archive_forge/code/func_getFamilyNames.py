import sys, os, pickle
from hashlib import md5
from xml.sax.saxutils import quoteattr
from time import process_time as clock
from reportlab.lib.utils import asBytes, asNative as _asNative
from reportlab.lib.utils import rl_isdir, rl_isfile, rl_listdir, rl_getmtime
def getFamilyNames(self):
    """Returns a list of the distinct font families found"""
    if not self._fontsByFamily:
        fonts = self._fonts
        for font in fonts:
            fam = font.familyName
            if fam is None:
                continue
            if fam in self._fontsByFamily:
                self._fontsByFamily[fam].append(font)
            else:
                self._fontsByFamily[fam] = [font]
    fsEncoding = self._fsEncoding
    names = list((asBytes(_, enc=fsEncoding) for _ in self._fontsByFamily.keys()))
    names.sort()
    return names