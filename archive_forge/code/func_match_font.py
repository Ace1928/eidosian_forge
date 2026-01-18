import os
import sys
import warnings
from os.path import basename, dirname, exists, join, splitext
from pygame.font import Font
def match_font(name, bold=False, italic=False):
    """pygame.font.match_font(name, bold=0, italic=0) -> name
    find the filename for the named system font

    This performs the same font search as the SysFont()
    function, only it returns the path to the TTF file
    that would be loaded. The font name can also be an
    iterable of font names or a string/bytes of comma-separated
    font names to try.

    If no match is found, None is returned.
    """
    initsysfonts()
    fontname = None
    if isinstance(name, (str, bytes)):
        name = name.split(b',' if isinstance(name, bytes) else ',')
    for single_name in name:
        if isinstance(single_name, bytes):
            single_name = single_name.decode()
        single_name = _simplename(single_name)
        styles = Sysfonts.get(single_name)
        if not styles:
            styles = Sysalias.get(single_name)
        if styles:
            while not fontname:
                fontname = styles.get((bold, italic))
                if italic:
                    italic = 0
                elif bold:
                    bold = 0
                elif not fontname:
                    fontname = list(styles.values())[0]
        if fontname:
            break
    return fontname