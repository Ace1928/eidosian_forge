from base64 import b64encode
from collections import namedtuple
import copy
import dataclasses
from functools import lru_cache
from io import BytesIO
import json
import logging
from numbers import Number
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Union
import matplotlib as mpl
from matplotlib import _api, _afm, cbook, ft2font
from matplotlib._fontconfig_pattern import (
from matplotlib.rcsetup import _validators
def findSystemFonts(fontpaths=None, fontext='ttf'):
    """
    Search for fonts in the specified font paths.  If no paths are
    given, will use a standard set of system paths, as well as the
    list of fonts tracked by fontconfig if fontconfig is installed and
    available.  A list of TrueType fonts are returned by default with
    AFM fonts as an option.
    """
    fontfiles = set()
    fontexts = get_fontext_synonyms(fontext)
    if fontpaths is None:
        if sys.platform == 'win32':
            installed_fonts = _get_win32_installed_fonts()
            fontpaths = []
        else:
            installed_fonts = _get_fontconfig_fonts()
            if sys.platform == 'darwin':
                fontpaths = [*X11FontDirectories, *OSXFontDirectories]
            else:
                fontpaths = X11FontDirectories
        fontfiles.update((str(path) for path in installed_fonts if path.suffix.lower()[1:] in fontexts))
    elif isinstance(fontpaths, str):
        fontpaths = [fontpaths]
    for path in fontpaths:
        fontfiles.update(map(os.path.abspath, list_fonts(path, fontexts)))
    return [fname for fname in fontfiles if os.path.exists(fname)]