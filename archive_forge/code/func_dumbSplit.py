from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ
from reportlab.lib.utils import isUnicode
import re
def dumbSplit(word, widths, maxWidths):
    """This function attempts to fit as many characters as possible into the available
    space, cutting "like a knife" between characters.  This would do for Chinese.
    It returns a list of (text, extraSpace) items where text is a Unicode string,
    and extraSpace is the points of unused space available on the line.  This is a
    structure which is fairly easy to display, and supports 'backtracking' approaches
    after the fact.

    Test cases assume each character is ten points wide...

    >>> dumbSplit(u'Hello', [10]*5, 60)
    [[10, u'Hello']]
    >>> dumbSplit(u'Hello', [10]*5, 50)
    [[0, u'Hello']]
    >>> dumbSplit(u'Hello', [10]*5, 40)
    [[0, u'Hell'], [30, u'o']]
    """
    _more = "\n    #>>> dumbSplit(u'Hello', [10]*5, 4)   # less than one character\n    #(u'', u'Hello')\n    # this says 'Nihongo wa muzukashii desu ne!' (Japanese is difficult isn't it?) in 12 characters\n    >>> jtext = u'日本語は難しいですね！'\n    >>> dumbSplit(jtext, [10]*11, 30)   #\n    (u'日本語', u'は難しいですね！')\n    "
    if not isinstance(maxWidths, (list, tuple)):
        maxWidths = [maxWidths]
    assert isUnicode(word)
    lines = []
    i = widthUsed = lineStartPos = 0
    maxWidth = maxWidths[0]
    nW = len(word)
    while i < nW:
        w = widths[i]
        c = word[i]
        widthUsed += w
        i += 1
        if widthUsed > maxWidth + _FUZZ and widthUsed > 0:
            extraSpace = maxWidth - widthUsed
            if ord(c) < 12288:
                limitCheck = lineStartPos + i >> 1
                for j in range(i - 1, limitCheck, -1):
                    cj = word[j]
                    if category(cj) == 'Zs' or ord(cj) >= 12288:
                        k = j + 1
                        if k < i:
                            j = k + 1
                            extraSpace += sum(widths[j:i])
                            w = widths[k]
                            c = word[k]
                            i = j
                            break
            if c not in ALL_CANNOT_START and i > lineStartPos + 1:
                i -= 1
                extraSpace += w
            lines.append([extraSpace, word[lineStartPos:i].strip()])
            try:
                maxWidth = maxWidths[len(lines)]
            except IndexError:
                maxWidth = maxWidths[-1]
            lineStartPos = i
            widthUsed = 0
    if widthUsed > 0:
        lines.append([maxWidth - widthUsed, word[lineStartPos:]])
    return lines