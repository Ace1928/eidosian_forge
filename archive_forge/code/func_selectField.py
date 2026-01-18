from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
def selectField(self, canvas, title, value, options, xmin, ymin, xmax, ymax):
    doc = canvas._doc
    page = doc.thisPageRef()
    R, G, B = obj_R_G_B(canvas._fillColorObj)
    font = canvas._fontname
    fontsize = canvas._fontsize
    field = SelectField(title, value, options, xmin, ymin, xmax, ymax, page, font=font, fontsize=fontsize, R=R, G=G, B=B)
    self.fields.append(field)
    canvas._addAnnotation(field)