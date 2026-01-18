from reportlab.pdfbase.pdfdoc import (PDFObject, PDFArray, PDFDictionary, PDFString, pdfdocEnc,
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import Color, CMYKColor, Whiter, Blacker, opaqueColor
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import isStr, asNative
import weakref
def radio(self, value=None, selected=False, buttonStyle='circle', shape='circle', fillColor=None, borderColor=None, textColor=None, borderWidth=1, borderStyle='solid', size=20, x=0, y=0, tooltip=None, name=None, annotationFlags='print', fieldFlags='noToggleToOff required radio', forceBorder=False, relative=False, dashLen=3):
    if name not in self._radios:
        group = RadioGroup(name, tooltip=tooltip, fieldFlags=fieldFlags)
        group._ref = self.getRef(group)
        self._radios[name] = group
        self.fields.append(group._ref)
    else:
        group = self._radios[name]
        fieldFlags = makeFlags(fieldFlags, fieldFlagValues)
        if fieldFlags != group.Ff:
            raise ValueError('radio.%s.%s created with different flags' % (name, value))
    if not value:
        raise ValueError('bad value %r for radio.%s' % (value, name))
    initialValue = value if selected else 'Off'
    textColor, borderColor, fillColor = self.stdColors(textColor, borderColor, fillColor)
    if initialValue == value:
        if group.V is not None:
            if group.V != value:
                raise ValueError('radio.%s.%s sets initial value conflicting with %s' % (name, value, group.V))
        else:
            group.V = value
    canv = self.canv
    if relative:
        x, y = self.canv.absolutePosition(x, y)
    doc = canv._doc
    AP = {}
    for key in 'NDR':
        APV = {}
        tC, bC, fC = self.varyColors(key, textColor, borderColor, fillColor)
        for v in (value, 'Off'):
            ap = self.checkboxAP(key, 'Yes' if v == value else 'Off', buttonStyle=buttonStyle, shape=shape, fillColor=fC, borderColor=bC, textColor=tC, borderWidth=borderWidth, borderStyle=borderStyle, size=size, dashLen=dashLen)
            if ap._af_refstr in self._refMap:
                ref = self._refMap[ap._af_refstr]
            else:
                ref = self.getRef(ap)
                self._refMap[ap._af_refstr] = ref
            APV[v] = ref
        AP[key] = PDFDictionary(APV)
        del APV
    RB = dict(FT=PDFName('Btn'), P=doc.thisPageRef(), AS=PDFName(initialValue), Rect=PDFArray((x, y, x + size, y + size)), AP=PDFDictionary(AP), Subtype=PDFName('Widget'), Type=PDFName('Annot'), F=makeFlags(annotationFlags, annotationFlagValues), Parent=group._ref, H=PDFName('N'))
    MK = dict(CA='(%s)' % ZDSyms[buttonStyle], BC=PDFArray(self.colorTuple(borderColor)), BG=PDFArray(self.colorTuple(fillColor)))
    if borderWidth:
        RB['BS'] = bsPDF(borderWidth, borderStyle, dashLen)
    RB['MK'] = PDFDictionary(MK)
    RB = PDFDictionary(RB)
    self.canv._addAnnotation(RB)
    group.kids.append(self.getRef(RB))
    self.checkForceBorder(x, y, size, size, forceBorder, shape, borderStyle, borderWidth, borderColor, fillColor)