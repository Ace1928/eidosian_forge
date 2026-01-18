import os
import tempfile
from io import BytesIO
from urllib.request import urlopen
from Bio.KEGG.KGML.KGML_pathway import Pathway
def color_to_reportlab(color):
    """Return the passed color in Reportlab Color format.

    We allow colors to be specified as hex values, tuples, or Reportlab Color
    objects, and with or without an alpha channel. This function acts as a
    Rosetta stone for conversion of those formats to a Reportlab Color
    object, with alpha value.

    Any other color specification is returned directly
    """
    if isinstance(color, colors.Color):
        return color
    elif isinstance(color, str):
        if color.startswith('0x'):
            color.replace('0x', '#')
        if len(color) == 7:
            return colors.HexColor(color)
        else:
            try:
                return colors.HexColor(color, hasAlpha=True)
            except TypeError:
                raise RuntimeError('Your reportlab seems to be too old, try 2.7 onwards') from None
    elif isinstance(color, tuple):
        return colors.Color(*color)
    return color