from reportlab.pdfbase.pdfdoc import (PDFObject, PDFArray, PDFDictionary, PDFString, pdfdocEnc,
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import Color, CMYKColor, Whiter, Blacker, opaqueColor
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import isStr, asNative
import weakref
def checkboxAP(self, key, value, buttonStyle='circle', shape='square', fillColor=None, borderColor=None, textColor=None, borderWidth=1, borderStyle='solid', size=20, dashLen=3):
    stream = [].append
    ds = size
    if shape == 'square':
        stream('q')
        streamFill = self.streamFillColor(fillColor)
        stream('1 g 1 G %(streamFill)s 0 0 %(size)s %(size)s re f')
        if borderWidth != None:
            streamStroke = self.streamStrokeColor(borderColor)
            hbw = borderWidth * 0.5
            smbw = size - borderWidth
            ds = smbw
            if borderStyle == 'underlined':
                stream('%(streamStroke)s %(borderWidth)s w 0 %(hbw)s m %(size)s %(hbw)s l s')
            elif borderStyle in ('dashed', 'inset', 'bevelled', 'solid'):
                if borderStyle == 'dashed':
                    dash = ' [%s ] 0 d' % fp_str(dashLen)
                else:
                    dash = ''
                stream('%(streamStroke)s%(dash)s %(borderWidth)s w %(hbw)s %(hbw)s %(smbw)s %(smbw)s re s')
            if borderStyle in ('bevelled', 'inset'):
                _2bw = 2 * borderWidth
                sm2bw = size - _2bw
                ds = sm2bw
                bbs0 = Blacker(fillColor, 0.5)
                bbs1 = fillColor
                if key != 'D':
                    bbs0, bbs1 = (bbs1, bbs0)
                bbs0 = self.streamFillColor(bbs0)
                bbs1 = self.streamFillColor(bbs1)
                stream('%(bbs0)s %(borderWidth)s %(borderWidth)s m %(borderWidth)s %(smbw)s l %(smbw)s %(smbw)s l %(sm2bw)s %(sm2bw)s l %(_2bw)s %(sm2bw)s l %(_2bw)s %(_2bw)s l f %(bbs1)s %(smbw)s %(smbw)s m %(smbw)s %(borderWidth)s l %(borderWidth)s %(borderWidth)s l %(_2bw)s %(_2bw)s l %(sm2bw)s %(_2bw)s l %(sm2bw)s %(sm2bw)s l f')
        stream('Q')
    elif shape == 'circle':
        cas = lambda _r, **_casKwds: self.circleArcStream(size, _r, **_casKwds)
        r = size * 0.5
        streamFill = self.streamFillColor(fillColor)
        stream('q 1 g 1 G %(streamFill)s')
        stream(cas(r))
        stream('f')
        stream('Q')
        if borderWidth != None:
            stream('q')
            streamStroke = self.streamStrokeColor(borderColor)
            hbw = borderWidth * 0.5
            ds = size - borderWidth
            if borderStyle == 'underlined':
                stream('q %(streamStroke)s %(borderWidth)s w 0 %(hbw)s m %(size)s %(hbw)s l s Q')
            elif borderStyle in ('dashed', 'inset', 'bevelled', 'solid'):
                if borderStyle == 'dashed':
                    dash = ' [3 ] 0 d'
                else:
                    dash = ''
                stream('%(streamStroke)s%(dash)s %(borderWidth)s w')
                stream(cas(r - hbw))
                stream('s')
            stream('Q')
            if borderStyle in ('bevelled', 'inset'):
                _3bwh = 3 * hbw
                ds = size - _3bwh
                bbs0 = Blacker(fillColor, 0.5)
                bbs1 = Whiter(fillColor, 0.5)
                a0 = (0, 1)
                a1 = (2, 3)
                if borderStyle == 'inset':
                    bbs0, bbs1 = (bbs1, bbs0)
                if key != 'D':
                    bbs0, bbs1 = (bbs1, bbs0)
                bbs0 = self.streamStrokeColor(bbs0)
                bbs1 = self.streamStrokeColor(bbs1)
                stream('q %(bbs0)s %(borderWidth)s w')
                stream(cas(r - _3bwh, rotated=True, arcs=a0))
                stream('S Q %(bbs1)s q')
                stream(cas(r - _3bwh, rotated=True, arcs=a1))
                stream('S Q')
    if value == 'Yes':
        textFillColor = self.streamFillColor(textColor)
        textStrokeColor = self.streamStrokeColor(textColor)
        stream('q %(textFillColor)s %(textStrokeColor)s')
        cbm = cbmarks[buttonStyle]
        if shape == 'circle' and buttonStyle == 'circle':
            stream(cas(max(r - (size - ds), 1) * 0.5))
            stream('f')
        else:
            stream(cbm.scaledRender(size, size - ds))
        stream('Q')
    stream = ('\n'.join(stream.__self__) % vars()).replace('  ', ' ').replace('\n\n', '\n')
    return self.makeStream(size, size, stream, Resources=PDFFromString('<< /ProcSet [/PDF] >>'))