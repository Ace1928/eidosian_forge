from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def buttonField(self, canvas, title, value, xmin, ymin, width=16.7704, height=14.907):
    doc = canvas._doc
    page = doc.thisPageRef()
    field = ButtonField(title, value, xmin, ymin, page, width=width, height=height)
    self.fields.append(field)
    canvas._addAnnotation(field)