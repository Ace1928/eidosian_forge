import os
import sys
import warnings
from os.path import basename, dirname, exists, join, splitext
from pygame.font import Font
def font_constructor(fontpath, size, bold, italic):
    """
    pygame.font specific declarations

    :param fontpath: path to a font.
    :param size: size of a font.
    :param bold: bold style, True or False.
    :param italic: italic style, True or False.

    :return: A font.Font object.
    """
    font = Font(fontpath, size)
    if bold:
        font.set_bold(True)
    if italic:
        font.set_italic(True)
    return font