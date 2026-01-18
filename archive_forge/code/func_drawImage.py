import re
import hashlib
from string import digits
from math import sin, cos, tan, pi
from reportlab import rl_config
from reportlab.pdfbase import pdfdoc
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen  import pathobject
from reportlab.pdfgen.textobject import PDFTextObject, _PDFColorSetter
from reportlab.lib.colors import black, _chooseEnforceColorSpace, Color, CMYKColor, toColor
from reportlab.lib.utils import ImageReader, isSeq, isStr, isUnicode, _digester, asUnicode
from reportlab.lib.rl_accel import fp_str, escapePDF
from reportlab.lib.boxstuff import aspectRatioFix
def drawImage(self, image, x, y, width=None, height=None, mask=None, preserveAspectRatio=False, anchor='c', anchorAtXY=False, showBoundary=False, extraReturn=None):
    """Draws the image (ImageReader object or filename) as specified.

        "image" may be an image filename or an ImageReader object. 
 
        x and y define the lower left corner of the image you wish to
        draw (or of its bounding box, if using preserveAspectRation below).
         
        If width and height are not given, the width and height of the
        image in pixels is used at a scale of 1 point to 1 pixel.  
       
        If width and height are given, the image will be stretched to fill 
        the given rectangle bounded by (x, y, x+width, y-height).  
        
        If you supply negative widths and/or heights, it inverts them and adjusts
        x and y accordingly.

        The method returns the width and height of the underlying image, since
        this is often useful for layout algorithms and saves you work if you have
        not specified them yourself.

        The mask parameter supports transparent backgrounds. It takes 6 numbers
        and defines the range of RGB values which will be masked out or treated
        as transparent.  For example with [0,2,40,42,136,139], it will mask out
        any pixels with a Red value from 0-2, Green from 40-42 and
        Blue from 136-139  (on a scale of 0-255).

        New post version 2.0:  drawImage can center an image in a box you
        provide, while preserving its aspect ratio.  For example, you might
        have a fixed square box in your design, and a collection of photos
        which might be landscape or portrait that you want to appear within 
        the box.  If preserveAspectRatio is true, your image will appear within
        the box specified.

        
        If preserveAspectRatio is True, the anchor property can be used to
        specify how images should fit into the given box.  It should 
        be set to one of the following values, taken from the points of
        the compass (plus 'c' for 'centre'):

                nw   n   ne
                w    c    e
                sw   s   se

        The default value is 'c' for 'centre'.  Thus, if you want your
        bitmaps to always be centred and appear at the top of the given box,
        set anchor='n'.      There are good examples of this in the output
        of test_pdfgen_general.py

        Unlike drawInlineImage, this creates 'external images' which
        are only stored once in the PDF file but can be drawn many times.
        If you give it the same filename twice, even at different locations
        and sizes, it will reuse the first occurrence, resulting in a saving
        in file size and generation time.  If you use ImageReader objects,
        it tests whether the image content has changed before deciding
        whether to reuse it.

        In general you should use drawImage in preference to drawInlineImage
        unless you have read the PDF Spec and understand the tradeoffs."""
    self._currentPageHasImages = 1
    if isinstance(image, ImageReader):
        rawdata = image.getRGBData()
        smask = image._dataA
        if mask == 'auto' and smask:
            mdata = smask.getRGBData()
        else:
            mdata = str(mask)
        if isUnicode(mdata):
            mdata = mdata.encode('utf8')
        name = _digester(rawdata + mdata)
    else:
        s = '%s%s' % (image, mask)
        if isUnicode(s):
            s = s.encode('utf-8')
        name = _digester(s)
    regName = self._doc.getXObjectName(name)
    imgObj = self._doc.idToObject.get(regName, None)
    if not imgObj:
        imgObj = pdfdoc.PDFImageXObject(name, image, mask=mask)
        imgObj.name = name
        self._setXObjects(imgObj)
        self._doc.Reference(imgObj, regName)
        self._doc.addForm(name, imgObj)
        smask = getattr(imgObj, '_smask', None)
        if smask:
            mRegName = self._doc.getXObjectName(smask.name)
            mImgObj = self._doc.idToObject.get(mRegName, None)
            if not mImgObj:
                self._setXObjects(smask)
                imgObj.smask = self._doc.Reference(smask, mRegName)
            else:
                imgObj.smask = pdfdoc.PDFObjectReference(mRegName)
            del imgObj._smask
    x, y, width, height, scaled = aspectRatioFix(preserveAspectRatio, anchor, x, y, width, height, imgObj.width, imgObj.height, anchorAtXY)
    self.saveState()
    self.translate(x, y)
    self.scale(width, height)
    self._code.append('/%s Do' % regName)
    self.restoreState()
    if showBoundary:
        self.drawBoundary(showBoundary, x, y, width, height)
    self._formsinuse.append(name)
    if extraReturn:
        for k in extraReturn.keys():
            extraReturn[k] = vars()[k]
    return (imgObj.width, imgObj.height)