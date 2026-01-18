import os
import sys
import warnings
from os.path import basename, dirname, exists, join, splitext
from pygame.font import Font
def initsysfonts_darwin():
    """Read the fonts on MacOS, and OS X."""
    fonts = {}
    fclist_locations = ['/usr/X11/bin/fc-list', '/usr/X11R6/bin/fc-list']
    for bin_location in fclist_locations:
        if exists(bin_location):
            fonts = initsysfonts_unix(bin_location)
            break
    if len(fonts) == 0:
        fonts = _font_finder_darwin()
    return fonts