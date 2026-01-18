from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ
from reportlab.lib.utils import isUnicode
import re
def getCharWidths(word, fontName, fontSize):
    """Returns a list of glyph widths.

    >>> getCharWidths('Hello', 'Courier', 10)
    [6.0, 6.0, 6.0, 6.0, 6.0]
    >>> from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    >>> from reportlab.pdfbase.pdfmetrics import registerFont
    >>> registerFont(UnicodeCIDFont('HeiseiMin-W3'))
    >>> getCharWidths(u'東京', 'HeiseiMin-W3', 10)   #most kanji are 100 ems
    [10.0, 10.0]
    """
    return [stringWidth(uChar, fontName, fontSize) for uChar in word]