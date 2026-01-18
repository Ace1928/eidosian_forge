from reportlab.pdfbase.pdfdoc import (PDFObject, PDFArray, PDFDictionary, PDFString, pdfdocEnc,
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import Color, CMYKColor, Whiter, Blacker, opaqueColor
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import isStr, asNative
import weakref
def checkbox(self, checked=False, buttonStyle='check', shape='square', fillColor=None, borderColor=None, textColor=None, borderWidth=1, borderStyle='solid', size=20, x=0, y=0, tooltip=None, name=None, annotationFlags='print', fieldFlags='required', forceBorder=False, relative=False, dashLen=3):
    initialValue = 'Yes' if checked else 'Off'
    textColor, borderColor, fillColor = self.stdColors(textColor, borderColor, fillColor)
    canv = self.canv
    if relative:
        x, y = self.canv.absolutePosition(x, y)
    doc = canv._doc
    AP = {}
    for key in 'NDR':
        APV = {}
        tC, bC, fC = self.varyColors(key, textColor, borderColor, fillColor)
        for value in ('Yes', 'Off'):
            ap = self.checkboxAP(key, value, buttonStyle=buttonStyle, shape=shape, fillColor=fC, borderColor=bC, textColor=tC, borderWidth=borderWidth, borderStyle=borderStyle, size=size, dashLen=dashLen)
            if ap._af_refstr in self._refMap:
                ref = self._refMap[ap._af_refstr]
            else:
                ref = self.getRef(ap)
                self._refMap[ap._af_refstr] = ref
            APV[value] = ref
        AP[key] = PDFDictionary(APV)
        del APV
    CB = dict(FT=PDFName('Btn'), P=doc.thisPageRef(), V=PDFName(initialValue), AS=PDFName(initialValue), Rect=PDFArray((x, y, x + size, y + size)), AP=PDFDictionary(AP), Subtype=PDFName('Widget'), Type=PDFName('Annot'), F=makeFlags(annotationFlags, annotationFlagValues), Ff=makeFlags(fieldFlags, fieldFlagValues), H=PDFName('N'))
    if tooltip:
        CB['TU'] = PDFString(tooltip)
    if not name:
        name = 'AFF%03d' % len(self.fields)
    if borderWidth:
        CB['BS'] = bsPDF(borderWidth, borderStyle, dashLen)
    CB['T'] = PDFString(name)
    MK = dict(CA='(%s)' % ZDSyms[buttonStyle], BC=PDFArray(self.colorTuple(borderColor)), BG=PDFArray(self.colorTuple(fillColor)))
    CB['MK'] = PDFDictionary(MK)
    CB = PDFDictionary(CB)
    self.canv._addAnnotation(CB)
    self.fields.append(self.getRef(CB))
    self.checkForceBorder(x, y, size, size, forceBorder, shape, borderStyle, borderWidth, borderColor, fillColor)