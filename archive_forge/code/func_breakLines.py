from reportlab.lib import PyFontify
from reportlab.platypus.paragraph import Paragraph, _handleBulletWidth, \
from reportlab.lib.utils import isSeq
from reportlab.platypus.flowables import _dedenter
def breakLines(self, width):
    """
        Returns a broken line structure. There are two cases

        A) For the simple case of a single formatting input fragment the output is
            A fragment specifier with
                - kind = 0
                - fontName, fontSize, leading, textColor
                - lines=  A list of lines
                
                    Each line has two items:
                    
                    1. unused width in points
                    2. a list of words

        B) When there is more than one input formatting fragment the out put is
            A fragment specifier with
                - kind = 1
                - lines =  A list of fragments each having fields:
                
                    - extraspace (needed for justified)
                    - fontSize
                    - words=word list
                    - each word is itself a fragment with
                    - various settings

        This structure can be used to easily draw paragraphs with the various alignments.
        You can supply either a single width or a list of widths; the latter will have its
        last item repeated until necessary. A 2-element list is useful when there is a
        different first line indent; a longer list could be created to facilitate custom wraps
        around irregular objects."""
    self._width_max = 0
    if not isSeq(width):
        maxWidths = [width]
    else:
        maxWidths = width
    lines = []
    lineno = 0
    maxWidth = maxWidths[lineno]
    style = self.style
    fFontSize = float(style.fontSize)
    requiredWidth = 0
    _handleBulletWidth(self.bulletText, style, maxWidths)
    self.height = 0
    autoLeading = getattr(self, 'autoLeading', getattr(style, 'autoLeading', ''))
    calcBounds = autoLeading not in ('', 'off')
    frags = self.frags
    nFrags = len(frags)
    if nFrags == 1:
        f = frags[0]
        if hasattr(f, 'text'):
            fontSize = f.fontSize
            fontName = f.fontName
            ascent, descent = getAscentDescent(fontName, fontSize)
            kind = 0
            L = f.text.split('\n')
            for l in L:
                currentWidth = stringWidth(l, fontName, fontSize)
                if currentWidth > self._width_max:
                    self._width_max = currentWidth
                requiredWidth = max(currentWidth, requiredWidth)
                extraSpace = maxWidth - currentWidth
                lines.append((extraSpace, l.split(' '), currentWidth))
                lineno = lineno + 1
                maxWidth = lineno < len(maxWidths) and maxWidths[lineno] or maxWidths[-1]
            blPara = f.clone(kind=kind, lines=lines, ascent=ascent, descent=descent, fontSize=fontSize)
        else:
            kind = f.kind
            lines = f.lines
            for L in lines:
                if kind == 0:
                    currentWidth = L[2]
                else:
                    currentWidth = L.currentWidth
                requiredWidth = max(currentWidth, requiredWidth)
            blPara = f.clone(kind=kind, lines=lines)
        self.width = max(self.width, requiredWidth)
        return blPara
    elif nFrags <= 0:
        return ParaLines(kind=0, fontSize=style.fontSize, fontName=style.fontName, textColor=style.textColor, ascent=style.fontSize, descent=-0.2 * style.fontSize, lines=[])
    else:
        for L in _getFragLines(frags):
            currentWidth, n, w = _getFragWord(L, maxWidth)
            f = w[0][0]
            maxSize = f.fontSize
            maxAscent, minDescent = getAscentDescent(f.fontName, maxSize)
            words = [f.clone()]
            words[-1].text = w[0][1]
            for i in w[1:]:
                f = i[0].clone()
                f.text = i[1]
                words.append(f)
                fontSize = f.fontSize
                fontName = f.fontName
                if calcBounds:
                    cbDefn = getattr(f, 'cbDefn', None)
                    if getattr(cbDefn, 'width', 0):
                        descent, ascent = imgVRange(imgNormV(cbDefn.height, fontSize), cbDefn.valign, fontSize)
                    else:
                        ascent, descent = getAscentDescent(fontName, fontSize)
                else:
                    ascent, descent = getAscentDescent(fontName, fontSize)
                maxSize = max(maxSize, fontSize)
                maxAscent = max(maxAscent, ascent)
                minDescent = min(minDescent, descent)
            lineno += 1
            maxWidth = lineno < len(maxWidths) and maxWidths[lineno] or maxWidths[-1]
            requiredWidth = max(currentWidth, requiredWidth)
            extraSpace = maxWidth - currentWidth
            if currentWidth > self._width_max:
                self._width_max = currentWidth
            lines.append(ParaLines(extraSpace=extraSpace, wordCount=n, words=words, fontSize=maxSize, ascent=maxAscent, descent=minDescent, currentWidth=currentWidth, preformatted=True))
        self.width = max(self.width, requiredWidth)
        return ParaLines(kind=1, lines=lines)
    return lines