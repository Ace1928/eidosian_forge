from collections import namedtuple
import enum
from functools import lru_cache, partial, wraps
import logging
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import numpy as np
from matplotlib import _api, cbook
@property
def font_path(self):
    """The `~pathlib.Path` to the font for this glyph."""
    psfont = self._get_pdftexmap_entry()
    if psfont.filename is None:
        raise ValueError('No usable font file found for {} ({}); the font may lack a Type-1 version'.format(psfont.psname.decode('ascii'), psfont.texname.decode('ascii')))
    return Path(psfont.filename)