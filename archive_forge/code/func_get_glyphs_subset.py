from io import BytesIO
import functools
from fontTools import subset
import matplotlib as mpl
from .. import font_manager, ft2font
from .._afm import AFM
from ..backend_bases import RendererBase
def get_glyphs_subset(fontfile, characters):
    """
    Subset a TTF font

    Reads the named fontfile and restricts the font to the characters.
    Returns a serialization of the subset font as file-like object.

    Parameters
    ----------
    fontfile : str
        Path to the font file
    characters : str
        Continuous set of characters to include in subset
    """
    options = subset.Options(glyph_names=True, recommended_glyphs=True)
    options.drop_tables += ['FFTM', 'PfEd', 'BDF', 'meta']
    if fontfile.endswith('.ttc'):
        options.font_number = 0
    with subset.load_font(fontfile, options) as font:
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(text=characters)
        subsetter.subset(font)
        fh = BytesIO()
        font.save(fh, reorderTables=False)
        return fh