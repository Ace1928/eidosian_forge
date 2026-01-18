import codecs
import datetime
from enum import Enum
import functools
from io import StringIO
import itertools
import logging
import os
import pathlib
import shutil
from tempfile import TemporaryDirectory
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib._afm import AFM
from matplotlib.backend_bases import (
from matplotlib.cbook import is_writable_file_like, file_requires_unicode
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LOAD_NO_SCALE, FT2Font
from matplotlib._ttconv import convert_ttf_to_ps
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps
def pstoeps(tmpfile, bbox=None, rotated=False):
    """
    Convert the postscript to encapsulated postscript.  The bbox of
    the eps file will be replaced with the given *bbox* argument. If
    None, original bbox will be used.
    """
    if bbox:
        bbox_info, rotate = get_bbox_header(bbox, rotated=rotated)
    else:
        bbox_info, rotate = (None, None)
    epsfile = tmpfile + '.eps'
    with open(epsfile, 'wb') as epsh, open(tmpfile, 'rb') as tmph:
        write = epsh.write
        for line in tmph:
            if line.startswith(b'%!PS'):
                write(b'%!PS-Adobe-3.0 EPSF-3.0\n')
                if bbox:
                    write(bbox_info.encode('ascii') + b'\n')
            elif line.startswith(b'%%EndComments'):
                write(line)
                write(b'%%BeginProlog\nsave\ncountdictstack\nmark\nnewpath\n/showpage {} def\n/setpagedevice {pop} def\n%%EndProlog\n%%Page 1 1\n')
                if rotate:
                    write(rotate.encode('ascii') + b'\n')
                break
            elif bbox and line.startswith((b'%%Bound', b'%%HiResBound', b'%%DocumentMedia', b'%%Pages')):
                pass
            else:
                write(line)
        for line in tmph:
            if line.startswith(b'%%EOF'):
                write(b'cleartomark\ncountdictstack\nexch sub { end } repeat\nrestore\nshowpage\n%%EOF\n')
            elif line.startswith(b'%%PageBoundingBox'):
                pass
            else:
                write(line)
    os.remove(tmpfile)
    shutil.move(epsfile, tmpfile)