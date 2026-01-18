from collections import OrderedDict
import logging
import urllib.parse
import numpy as np
from matplotlib import _text_helpers, dviread
from matplotlib.font_manager import (
from matplotlib.ft2font import LOAD_NO_HINTING, LOAD_TARGET_LIGHT
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
def get_text_path(self, prop, s, ismath=False):
    """
        Convert text *s* to path (a tuple of vertices and codes for
        matplotlib.path.Path).

        Parameters
        ----------
        prop : `~matplotlib.font_manager.FontProperties`
            The font properties for the text.
        s : str
            The text to be converted.
        ismath : {False, True, "TeX"}
            If True, use mathtext parser.  If "TeX", use tex for rendering.

        Returns
        -------
        verts : list
            A list of arrays containing the (x, y) coordinates of the vertices.
        codes : list
            A list of path codes.

        Examples
        --------
        Create a list of vertices and codes from a text, and create a `.Path`
        from those::

            from matplotlib.path import Path
            from matplotlib.text import TextToPath
            from matplotlib.font_manager import FontProperties

            fp = FontProperties(family="Comic Neue", style="italic")
            verts, codes = TextToPath().get_text_path(fp, "ABC")
            path = Path(verts, codes, closed=False)

        Also see `TextPath` for a more direct way to create a path from a text.
        """
    if ismath == 'TeX':
        glyph_info, glyph_map, rects = self.get_glyphs_tex(prop, s)
    elif not ismath:
        font = self._get_font(prop)
        glyph_info, glyph_map, rects = self.get_glyphs_with_font(font, s)
    else:
        glyph_info, glyph_map, rects = self.get_glyphs_mathtext(prop, s)
    verts, codes = ([], [])
    for glyph_id, xposition, yposition, scale in glyph_info:
        verts1, codes1 = glyph_map[glyph_id]
        verts.extend(verts1 * scale + [xposition, yposition])
        codes.extend(codes1)
    for verts1, codes1 in rects:
        verts.extend(verts1)
        codes.extend(codes1)
    if not verts:
        verts = np.empty((0, 2))
    return (verts, codes)