from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def getForm(canvas):
    """get form from canvas, create the form if needed"""
    try:
        return canvas.AcroForm
    except AttributeError:
        theform = canvas.AcroForm = AcroForm()
        d = canvas._doc
        cat = d._catalog
        cat.AcroForm = theform
        return theform