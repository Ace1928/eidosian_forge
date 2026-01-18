import sys
import os
import pygame as pg
def get_font_list(self):
    """
        Generate a font list using font.get_fonts() for system fonts or
        from a path from the command line.
        """
    path = ''
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        path = os.path.join(sys.argv[1], '')
    fonts = []
    if os.path.exists(path):
        for font in os.listdir(path):
            if font.endswith('.ttf'):
                fonts.append(font)
    return (fonts or pg.font.get_fonts(), path)